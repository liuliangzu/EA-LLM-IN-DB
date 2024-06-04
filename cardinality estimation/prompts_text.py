prompt_template = (
        "You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )

prompt_startword = (
    "You are a DBMS, you should use query to finish the Cardinal_estimation task, now the information will be listed:\n"
)
prompt_column = (
    "the table and column name : {table_name}\n"
    "----this column max value : {max_value}\n"
    "----this column min value : {min_value}\n"
    "----this column cardinality : {cardinality}\n"
    "----this column unique nums : {num_unique_values}\n"
)
prompt_index = (
    "the index column : {index_column_name}\n"
    "----index type : {index_type}\n"
    "----index column max value: {index_max}\n"
    "----index column min value: {index_min}\n"
    "----index column cardinality : {cardinality}\n"
    "----index column unique nums : {num_unique_values}\n"
)
prompt_query_predict = (
    "the query predict is: {predict}"
)
prompt_query_join_operator = (
    "the query join operator is: {join_operator}"
)
