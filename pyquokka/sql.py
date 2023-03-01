from pyquokka.sql_utils import evaluate, required_columns_from_exp, is_cast_to_date
import polars
import sqlglot
from sqlglot import expressions as exp
from sqlglot.optimizer import optimize
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.optimizer.simplify import simplify
from pyquokka.sql_utils import evaluate, required_columns_from_exp, is_cast_to_date

from collections import deque # for toposorting of cte

agg_types = {
        exp.Sum: "sum",
        exp.Avg: "avg",
        exp.Count: "count",
        exp.Min: "min",
        exp.Max: "max"
}

def add_filter(d, f):
    """
    Args:
        d: Polars dataframe or Quokka datastream
        f: SQLGlot expression for filter
    
    Returns d with filter f applied.
    """
    if isinstance(d, polars.internals.dataframe.frame.DataFrame):
        d = d.filter(evaluate(f))
    else: 
        d = d.filter(f.sql())

    return d

def add_column(d, e, alias):
    """
    Args:
        d: Polars dataframe or Quokka datastream
        e: SQLGlot expression for new column
        alias: string
        
    Returns d with new column derived from expression e, with alias as the new column's name. 
    """
    if isinstance(d, polars.internals.dataframe.frame.DataFrame):
        d = d.with_column(evaluate(e)(d).alias(alias))
    else:
        d = d.with_column(str(alias), evaluate(e), required_columns=required_columns_from_exp(e))
#         print(d.schema)
#         print(alias)
#         d = d.with_column_sql(e.sql() + " as " + alias)

    return d

def join_tables(d1, d2, key1, key2, join_index, query, how = 'inner'):
    """
    Args:
        d1: Polars dataframe or Quokka datastream
        d2: Polars dataframe or Quokka datastream
        key1 (exp.Column): column in d1
        key2 (exp.Column): column in d2
        join_index (int): Used to determine suffix to use for join
        query (exp): SQLGlot expression representing original query
        how (string): 'inner', 'left', etc.
    
    Returns (joined, query):
        joined: the result of joining d1 and d2 on key1 = key2, which is either a Polars dataframe or Quokka datastream.
        query: rewritten query reflecting any column name changes resulting from joins.
        
    Currently running into bugs when the query is rewritten so the returned query is the same for now.
    """
    suffix = '_' + str(join_index + 2)
    switched_order = False
    if isinstance(d1, polars.internals.dataframe.frame.DataFrame) and not isinstance(d2, polars.internals.dataframe.frame.DataFrame):
        # Need to switch order 
        joined = d2.join(d1, left_on = key2.name, right_on = key1.name, suffix=suffix, how=how)
        switched_order = True
    else:
        joined = d1.join(d2, left_on = key1.name, right_on = key2.name, suffix=suffix, how=how)
    
    # If columns that have been renamed due to the join appear in the query, then rewrite the query to reflect the change. 
    
    renamed = dict() # old name: new name
    for c in query.find_all(exp.Column):
        if switched_order:
            for col in d1.schema:
                if col in d2.schema:
                    renamed[col] = col + suffix
        else:
            for col in d2.schema:
                if col in d1.schema: 
                    renamed[col] = col + suffix
                    
    for c in query.find_all(exp.Column):
        if switched_order and c.table == key1.table and c.name in renamed:
            c.replace(sqlglot.parse_one(key1.table + "." + renamed[c.name]))
        elif c.table == key2.table and c.name in renamed:
            c.replace(sqlglot.parse_one(key2.table + "." + renamed[c.name]))
    
    # Add right column back
    if key1.name == key2.name:
        return joined, query
    elif isinstance(joined, polars.internals.dataframe.frame.DataFrame):
        # This only happens when both d1 and d2 are polars dataframe, so we didn't switch the order
        if key2.name not in joined.schema:
            joined = joined.with_column(polars.col(key1.name).alias(key2.name))
    else:
        if switched_order and key1.name not in joined.schema:
            joined = joined.with_column(key1.name, lambda x: x[key2.name], required_columns = {key2.name})
        elif key2.name not in joined.schema:
            joined = joined.with_column(key2.name, lambda x: x[key1.name], required_columns = {key1.name})
    
    return joined, query
        
def add_joins(d, tables, joins, query, v=False):
    """
    Args:
        d: Quokka datastream or Polars dataframe
        tables (dict): mapping of name (str) to table
        joins (list): list of SQLGlot join expressions
        query (exp): SQLGlot expression representing original query
        v (bool): verbose option
    
    Applies each join in list of joins to d. 
    """
    
    # for aliasing in joins, don't need changes to persist beyond this function and don't want to overwrite existing entries
    tables_copy = tables.copy()
    
    joined_tables = dict() # key: table, value: [df it has been joined to, all tables in that df]
    filters = []
    join_index = 0

    for join in joins:
        if join.this.name != join.this.alias_or_name:
            if v: print("aliased: ", join.this.name, join.this.alias_or_name)
            if join.this.name not in tables:
                tables_copy[join.this.name] = tables_copy[join.this.alias_or_name]
            else:
                tables_copy[join.this.alias_or_name] = tables_copy[join.this.name]
                
        how, left, right, remaining_condition = join_condition(join)
        if remaining_condition: filters.append(remaining_condition)
        for i in range(len(left)):
            lkey = left[i]; rkey = right[i]
            ltable = lkey.table; rtable = rkey.table
            
            # If neither table has been joined to a table before: join the two tables
            if ltable not in joined_tables and rtable not in joined_tables:
                # join ltable and rtable on lkey = rkey
                if v: print("join " + ltable + ", " + rtable + " on ", lkey.name, rkey.name)
                joined, query = join_tables(tables_copy[ltable], tables_copy[rtable], lkey, rkey, join_index, query, how=how)
                join_index += 1
                
                joined_tables[ltable] = [joined, [ltable, rtable]]
                joined_tables[rtable] = [joined, [ltable, rtable]]
            
            # If one table has been joined to a table before: join the other to that group
            elif ltable in joined_tables and rtable not in joined_tables:
                if v: print("join ", joined_tables[ltable][1], ", " + rtable + " on ", lkey.name, rkey.name)
                
                l = joined_tables[ltable][0]
                joined, query = join_tables(l, tables_copy[rtable], lkey, rkey, join_index, query, how=how)
                join_index += 1
            
                joined_tables[ltable][0] = joined
                joined_tables[ltable][1].append(rtable)
                
                for table in ([rtable] + joined_tables[ltable][1]):
                    joined_tables[table] = joined_tables[ltable]
                
            elif ltable not in joined_tables and rtable in joined_tables:
                if v: print("join ", ltable, ", ", joined_tables[rtable][1], " on ", lkey.name, rkey.name)
            
                r = joined_tables[rtable][0]
                joined, query = join_tables(tables_copy[ltable], r, lkey, rkey, join_index, query, how=how)
                join_index += 1
                
                joined_tables[rtable][0] = joined
                joined_tables[rtable][1].append(ltable)
                
                for table in ([ltable] + joined_tables[rtable][1]):
                    joined_tables[table] = joined_tables[rtable]
            else:
                # If both tables have been joined but to separate groups, join the groups
                if joined_tables[ltable][1] != joined_tables[rtable][1]:
                    if v: print("join ", joined_tables[ltable][1], ", ", joined_tables[rtable][1], " on ", lkey.name, rkey.name)
                    
                    l = joined_tables[ltable][0]; r = joined_tables[rtable][0]
                    joined, query = join_tables(l, r, lkey, rkey, join_index, query, how=how)
                    join_index += 1
                    
                    group1 = joined_tables[ltable][1]; group2 = joined_tables[rtable][1]
                    for table in (group1 + group2):
                        joined_tables[table] = [joined, group1 + group2]
                    
                # If both tables have been joined to the same group, apply as filter
                else:
                    if v: print("filter ", lkey.name + " = " + rkey.name)
                    filters.append(sqlglot.parse_one(lkey.name + " = " + rkey.name))

    result = joined_tables.popitem()[1][0]
    if v: print("filters from joins: ")
    for f in filters:
        if v: print(f)
        result = add_filter(result, f)
    return result, query

def evaluate_CTE(query, tables, v=False):
    """
    Args:
        query (str): SQL query
        tables (dict): mapping of name (str) to table (either polars dataframe or Quokka datastream)
        v (bool): verbose option
    """
    if v: print("\n QUERY: \n", query.sql(pretty=True), "\n -----")
    from_name = query.args.get('from').find(sqlglot.expressions.Table).name
    d = tables[from_name]
    
    if v: print("Initializing dataframe to ", from_name)

    joins = query.args.get('joins')

    if joins:
        d, query = add_joins(d, tables, joins, query, v=v)
        if v: print("schema after joins: ", d.schema, "\n -----")
    if v: print("rewritten query after joins: \n", query.sql(pretty=True), "\n -----")
                
    where = query.args.get('where')
    if where:
        for f in where.flatten():
            d = add_filter(d, f)
        if v: print("filters: " + where.sql() + "\n -----")
    
    selected = []
    rename_dict = dict()
    aggregations = []
    for e in query.expressions:
        aggs = [a for a in e.find_all(exp.AggFunc)]
        if aggs:
            if not e.alias:
                alias = name_agg(e, d.schema)
                aggregations.append(e.sql() + " as " + alias)
            else:
                alias = e.alias
                aggregations.append(e.sql())
        else:
            if isinstance(e, exp.Column) or isinstance(e.unalias(), exp.Column):
                if e.alias and e.unalias().name != e.alias:
                    if v: print("Alias ", e.unalias().name, " as ", e.alias)
                    rename_dict[e.unalias().name] = e.alias
                selected.append(e.unalias().name)
            else:
                d = add_column(d, e.unalias(), e.alias)
                selected.append(e.alias)

    if selected and not aggregations:
        if v: print("No aggregations; projections: " + str(selected) + "\n -----")
        d = d.select(selected)
    
    group = query.args.get('group')
    order = query.args.get('order')
    if group:
        groupby_cols = []
        index = 0
        for g in group.flatten():
            if isinstance(g, exp.Column):
                groupby_cols.append(g.name)
            else:
                groupcol_name = "_group" + str(index)
                d = add_column(d, g, groupcol_name)
                groupby_cols.append(groupcol_name)
                index += 1
                selected.append(groupcol_name)
        if order: 
            asc_or_desc = lambda x: 'desc' if x is True else 'asc'
            orderby_cols = [(o.this.name, asc_or_desc(o.args.get('desc'))) for o in order.flatten() if o.this.name in groupby_cols]
            print("orderby " + str(orderby_cols) + "\n -----")
            
            # For now, skip orderby if the column names aren't a subset of groupby_cols
            if len(orderby_cols) == 0:
                d = d.groupby(groupby_cols).agg_sql(','.join(aggregations))
            else:
                d = d.groupby(groupby_cols, orderby=orderby_cols).agg_sql(','.join(aggregations))
        else: 
            d = d.groupby(groupby_cols).agg_sql(','.join(aggregations))
        
        if v: print("groupby " + str(groupby_cols) + "\n -----")
        if v: print("agg: " + str(aggregations) + "\n -----")
    else:
        if aggregations:
            if v: print("agg: " + str(aggregations) + "\n -----")
            d = d.agg_sql(','.join(aggregations))
    
    if rename_dict:
        if v: print("rename dict: ", str(rename_dict), "\n -----")
        d = d.rename(rename_dict)
    if v: print("END CTE")
    return d

class CTE:
    def __init__(self, cte, tables):
        self.name = cte.alias
        self.expr = cte.this
        all_deps = [i.name for i in cte.find_all(exp.Table)]
        self.dependencies = [name for name in all_deps if name not in tables]
        #self.is_blocking = (self.expr.args.get('group') is not None)
        
        # If the expression is a single aggregation with no groupbys it's just a number; calculate this number and rewrite query with the column as a constant
        # TODO: adapt this to extend to multiple aggregations with no groupbys.
        self.is_scalar = (len(self.expr.expressions) == 1 and self.expr.find(exp.AggFunc) is not None and self.expr.args.get('group') is None)
        if self.is_scalar:
            self.scalar_name = self.expr.expressions[0].alias_or_name
def toposort_ctes(ctes):
    cte_aliases = {t.name: t for t in ctes}
    graph = {t.name: t.dependencies for t in ctes}
    result = deque()
    visited = set()

    stack = [[key for key in graph]]

    while stack:
      for i in stack[-1]: 
        if i in visited and i not in result: 
          result.appendleft(i)
        if i not in visited:
          visited.add(i)
          stack.append(graph[i]) 
          break
      else: 
        stack.pop() 

    names = list(result)[::-1]
    return [cte_aliases[name] for name in names]

def plan(query, tables, v=False):
    with_tables = query.args.get('with')
    if not with_tables: return evaluate_CTE(query, tables, v)

    ctes = []
    for cte in with_tables.flatten():
        ctes.append(CTE(cte, tables))
    order = toposort_ctes(ctes)
    for t in order:
        tables[t.name] = evaluate_CTE(t.expr, tables, v)
        if t.is_scalar:
            a = tables[t.name].collect().item()
            if v: print(t.name, "is a scalar:", a)
            query = replace_table_with_scalar(query, t.name, t.scalar_name, a)
    result = evaluate_CTE(query, tables, v)
    return result

def replace_table_with_scalar(query, table_name, scalar_name, a):
    """
    Remove table references from the query and replace any column references with the scalar a. 
    Args:
        query (sqlglot expression): SQL query
        name (str): table name
        a (numeric type): value of table as a scalar
    Returns:
        query (str): modified query
    """
    # Replace references to column in table with the scalar
    for c in query.find_all(exp.Column):
        if c.table == table_name or c.name == scalar_name:
            c.replace(sqlglot.parse_one(str(a)))
    return query
    
def name_agg(e, schema=None):
    """
    If an alias for an aggregation is not provided, generate a descriptive name that doesn't conflict with the given schema. Column will be named after the aggregation, if there is none then a generic name will be returned.
    Args:
        e (sqlglot expression): column with an aggregation. 
    Returns:
        alias (string): descriptive name for the column
    """
    if e.find(exp.AggFunc):
        a = e.find(exp.AggFunc)
        i = 0
        agg_name = agg_types[type(a)]
        while agg_name + '_' + str(i) in schema:
            i += 1
        return (agg_name + '_' + str(i))
    else:
        i = 0
        while 'col_' + str(i) in schema:
            i += 1
        return ('col_' + str(i))
    
# sqlglot.optimizer.eliminate_joins with some patches
def join_condition(join):
    """
    Extract the join condition from a join expression.
    Args:
        join (exp.Join)
    Returns:
        tuple[list[str], list[str], exp.Expression]:
            Tuple of (source key, join key, remaining predicate)
    """
    how = 'inner'
    if join.side == 'LEFT':
        how = 'left'
    if join.kind == 'CROSS':
        how = 'cross'
        return 'cross', join.this, None, join.args.get("on")
    
    name = join.this.alias_or_name
    on = join.args.get("on") or exp.TRUE
    on = on.copy()
    source_key = []
    join_key = []

    # find the join keys
    # SELECT
    # FROM x
    # JOIN y
    #   ON x.a = y.b AND y.b > 1
    #
    # should pull y.b as the join key and x.a as the source key
    if normalized(on):
        for condition in on.flatten() if isinstance(on, exp.And) else [on]:
            if isinstance(condition, exp.EQ):
                left, right = condition.unnest_operands()
                left_tables = exp.column_table_names(left)
                right_tables = exp.column_table_names(right)
                
                if not (left_tables and right_tables):
                    # This is a filter, should be in remaining_condition 
                    continue
                if (name in left_tables and name not in right_tables):
                    join_key.append(left)
                    source_key.append(right)
                    condition.replace(exp.TRUE)
                    if not isinstance(on, exp.And): on = exp.TRUE
                elif name in right_tables and name not in left_tables:
                    join_key.append(right)
                    source_key.append(left)
                    condition.replace(exp.TRUE)
                    if not isinstance(on, exp.And): on = exp.TRUE

    on = simplify(on)
    remaining_condition = None if on == exp.TRUE else on
    assert(len(source_key) == len(join_key))

    return how, source_key, join_key, remaining_condition