import copy
import datasets
import pandas as pd
import json
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer,TextStreamer
#import psycopg2 as p2
# example data preprocess
card_join={
    't.id=ci.movie_id':36244344,
    't.id=mc.movie_id':2609129,
    't.id=mk.movie_id':4523930,
    't.id=mi_idx.movie_id':1380035,
    't.id=mi.movie_id':14835720
}
# mi_idx.movie_id,mk.movie_id,ci.movie_id,mc.movie_id
column_statistic_info = pd.read_csv('/mnt/bd/llama-finetune/llama3/llama3_test/learnedcardinalities/data/column_min_max_vals.csv',delimiter=',')

prompt_template = (
        "You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )
prompt_samples = (
    "This query is sampled and {x} samples out of 1000 tuples are found to match the query conditions "
)
prompt_startword = (
    "You are a DBMS, you should use query to finish the Cardinal_estimation task, now the information will be listed: "
)
prompt_table = (
    "the number {times} table name : {table_name} and called {table_name_sx} "
)

prompt_query_predict = (
    "the number {times} query predict will be listed:  column is : {column_name},  predicate operator is : {operator},  predicate value is : {value} , and the cardinality of this predicate is {card}  "
)
prompt_query_join_operator = (
    "join column is : {column_name} and {column_name_2} , {column_name} is : {key_1} and index {index_1} , {column_name_2} is : {key_2} and index {index_2}, and the cardinality of this join is {card} "
)
promp_column_Cardinality = (
    "the column {name} information max_val = {maxv}, min_val = {minv}, cardinality = {card}, num_unique_values = {unique}  "
)
promp_most = (
    ",most_common_values: {mcv} ,most_common_freq {mcf} "
)
promp_hist = (
    ",histogram_bounds: {hist}"
)
prompt_pg_ce = (
    "The postgresql database gives an estimate of {pg_ce} for this query and other deep learning estimators give an estimate of {pr_ce}"
)
prompt_answer = (
    "---Answer is:  "
)
'''
假设无索引测试
prompt_index = (
    "the index column : {index_column_name}\n"
    "----index type : {index_type}\n"
    "----index column max value: {index_max}\n"
    "----index column min value: {index_min}\n"
    "----index column cardinality : {cardinality}\n"
    "----index column unique nums : {num_unique_values}\n"
)'''
'''pg_stats_ = "select most_common_vals,most_common_freqs from pg_stats where tablename = '{}' and attname = '{}'"
conn = p2.connect()'''
'''hard_column_name = 't.production_year'
most_common_vals = [2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2013,2001,2000,1999,1998,1997,1996,1995,1994,1992,1993,1990,1991,1987,1989,1985,1988,1986,1966,1982,1978,1973,1980,1981,1965,1970,1976,1983,1979,1967,1971,1984,1975,1969,1977,1968,1974,1972,1961,1963,1962,1964,1959,1960,1955,1957,1958,1915,1913,1914,1953,1956,1911,1954,1912,1952,1951,1916,1909,1949,1917,1910,1920,1922,1939,1919,1918,1950,2014,1921,1935,1924,1930,1903,1938,1940,1926,1928,1933,1941,1932,1925,1929,1934,1948,1927,1944,1931,1937]
most_common_freqs = [0.06656667,0.06273333,0.056633335,0.050066665,0.049133334,0.048433334,0.0434,0.0372,0.031733334,0.026666667,0.025933333,0.025633333,0.0236,0.021933334,0.0209,0.018633334,0.015733333,0.015266667,0.0137,0.012366666,0.010233333,0.0097,0.009266667,0.009133333,0.008266667,0.0079666665,0.0077333334,0.0075,0.0066333334,0.0063,0.006,0.005933333,0.0059,0.0059,0.0058333334,0.0057,0.0056666667,0.0056666667,0.0056666667,0.005566667,0.0055333334,0.0055,0.0054666665,0.0053666667,0.0053,0.005266667,0.005133333,0.005,0.004833333,0.0047333334,0.0047333334,0.0046333335,0.0044,0.0039333333,0.0038666667,0.0034,0.0034,0.0033666666,0.0033,0.0031666667,0.0031666667,0.0031666667,0.0031,0.0030333332,0.0030333332,0.0029666666,0.0026666666,0.0024666667,0.0020333333,0.0019666667,0.0018333333,0.0018,0.0017,0.0015666666,0.0014333334,0.0014333334,0.0013666666,0.0013333333,0.0013333333,0.0012666667,0.0012333334,0.0012,0.0011666666,0.0011666666,0.0011333333,0.0011333333,0.0011,0.0010666667,0.0010666667,0.0010666667,0.0010666667,0.0010333334,0.0009666667,0.0009666667,0.0009666667,0.0009666667,0.00093333336,0.00093333336,0.0009,0.0009]

prompt_hard_column = (
    "there is a hard column in this query names = {}\n"
    "----the column most common vals = {}\n"
    "----the column most common freqs = {}\n"
).format(hard_column_name,most_common_vals,most_common_freqs)'''

def get_preprocessed_arithmetic(dataset_config, tokenizer, split):
    if split=="train":
        data_path = "arithmetic_data/arithmetic_train.csv"
    elif split=="validation":
        data_path = "arithmetic_data/arithmetic_validation.csv"
    elif split=="test":
        data_path = "arithmetic_data/arithmetic_test.csv"

    dataset = datasets.load_dataset(
        "csv", 
        data_files={split: "arithmetic_data/arithmetic_train.csv"}
        )[split]


    prompt = (
        f"Calculate the following expression:\n{{instruction}}\n---\nAnswer:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(instruction=sample["instruction"]),
            "output": sample["output"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["output"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

# finetune query generate
def get_preprocessed_query(dataset_config, tokenizer, splits):
    if splits == "train":
        print("start train_data preprocess!")
    elif splits=="test":
        print("start test_data preprocess!")
    dataset = datasets.load_dataset(
        "csv", 
        data_files={splits: './learnedcardinalities/hf_set.csv'}
        )[splits]
    '''
    prompt = (
        f"You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n{{table}}\n---the join_operater is:\n{{operater}}\n---the predicate is:\n{{predicate}}\n---\nAnswer:\n"
    )
    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(table=sample["TableName_or_ColumnName"],operater=sample["join_operater"],predicate=sample["predicate"]),
            "output": str(sample["Cardinal_estimation_results"]),
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    '''
    dist = {
        't':3,
        'mc':4,
        'ci':4,
        'mi':3,
        'mi_idx':3,
        'mk':3
    }
    def apply_prompt_template(sample):
        prompt = prompt_startword
        column_list = str(sample["TableName_or_ColumnName"]).split(",")
        for i in range(len(column_list)):
            column_name_sx = column_list[i].split(" ")[1]
            column_name = column_list[i].split(" ")[0]
            prompt = prompt + prompt_table.format(times=i, table_name=column_name, table_name_sx = column_name_sx, column_num =dist[column_name_sx])
        join_list = str(sample["join_operater"]).split(",")
        column_set = list()
        for i in range(len(join_list)):
            join = join_list[i]
            join = join.split("=")
            column_1 = join[0]
            column_2 = join[1]
            column_set.append(column_1)
            column_set.append(column_2)
            if column_1.split(".")[1]=="id":
                key_1 = "Primary Key"
                index_1 = "None"
            elif column_1.split(".")[1].split("_")[-1]=="id":
                key_1 = "Foreign Key"
                index_1 = "None"
            else:
                key_1 = "Data Column"
                index_1 = "None"
            if column_2.split(".")[1]=="id":
                key_2 = "Primary Key"
                index_2 = "None"
            elif column_2.split(".")[1].split("_")[-1]=="id":
                key_2 = "Foreign Key"
                index_2 = "None"
            else:
                key_2 = "Data Column"
                index_2 = "None"
            prompt = prompt + prompt_query_join_operator.format(column_name = column_1,column_name_2 = column_2,key_1=key_1,index_1=index_1,key_2=key_2,index_2=index_2)
            
        predict_list = str(sample["predicate"]).split(",")
        i = 0
        j = 1
        while i < len(predict_list) and len(predict_list) % 3 == 0:
            prompt = prompt + prompt_query_predict.format(times = j,column_name=predict_list[i],operator=predict_list[i+1],value=predict_list[i+2])
            column_set.append(predict_list[i])
            #column_info = column_statistic_info[column_statistic_info["name"]==predict_list[i]]
            #prompt = prompt + promp_column_Cardinality.format(name=predict_list[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            j = j + 1
            i = i+3
        if len(predict_list) == 1:
            prompt = prompt + "no predicate for this query \n"
        column_set = list(set(column_set))
        for column in column_set:
            column_info = column_statistic_info[column_statistic_info["name"]==column]
            prompt = prompt + promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
        
        prompt = prompt + prompt_answer
        return {
            "prompt": str(prompt),
            "output": str(sample["Cardinal_estimation_results"]),
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["output"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

import psycopg2 as p2
# generate dataset as json
def generate_Dataset(splits):
    '''a = pd.read_csv('./learnedcardinalities-master/workloads/synthetic.csv',delimiter='#',header = None,names=['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results','index'])
    a = a.drop(columns=['index'])
    a = a[a['join_operater'].isnull()]
    a.to_csv('./synthetic_onetable.csv',index=False)'''
    file_path = './output_synthetic.sql'
    with open(file_path, 'r') as f:
        lines_sql = f.readlines()
    query_workload = [lines.split('||')[0] for lines in lines_sql]
    pg_ce = [lines.split('||')[2] for lines in lines_sql]
    pr_ce = [lines.split('||')[3].replace('\n','') for lines in lines_sql]
    conn_totaltable = p2.connect(
        host="localhost",
        database="template1",
        user="pilotscope",
        password="pilotscope",
        port=5432
    )
    conn_sample_info = p2.connect(
        host="localhost",
        database="sample_imdb",
        user="pilotscope",
        password="pilotscope",
        port=5432
    )
    cur_all = conn_totaltable.cursor()
    cur_sample = conn_sample_info.cursor()
    dataset = pd.read_csv('/mnt/bd/llama-finetune/llama3/llama3_test/learnedcardinalities/workloads/synthetic.csv',delimiter='#',header = None,names=['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results','index'])
    dataset["query"] = query_workload
    dataset["pg_ce"] = pg_ce
    dataset["pr_ce"] = pr_ce
    print("loaded \n")
    print(dataset.head(5))
    dist = {
        't':3,
        'mc':4,
        'ci':4,
        'mi':3,
        'mi_idx':3,
        'mk':3
    }
    def apply_prompt_template(sample):
        prompt = ""
        input_dict = {}
        column_list = str(sample["TableName_or_ColumnName"]).split(",")
        for i in range(len(column_list)):
            column_name_sx = column_list[i].split(" ")[1]
            column_name = column_list[i].split(" ")[0]
            input_dict["table_{} information".format(i)] = prompt_table.format(times=i, table_name=column_name, table_name_sx = column_name_sx)
        join_list = str(sample["join_operater"]).split(",")
        column_set = list()
        for i in range(len(join_list)):
            join = join_list[i]
            join = join.split("=")
            if len(join) < 2:
                break
            column_1 = join[0]
            column_2 = join[1]
            column_set.append(column_1)
            column_set.append(column_2)
            if column_1.split(".")[1]=="id":
                key_1 = "Primary Key"
                index_1 = "None"
            elif column_1.split(".")[1].split("_")[-1]=="id":
                key_1 = "Foreign Key"
                index_1 = "None"
            else:
                key_1 = "Data Column"
                index_1 = "None"
            if column_2.split(".")[1]=="id":
                key_2 = "Primary Key"
                index_2 = "None"
            elif column_2.split(".")[1].split("_")[-1]=="id":
                key_2 = "Foreign Key"
                index_2 = "None"
            else:
                key_2 = "Data Column"
                index_2 = "None"
            input_dict["join_{} information".format(i)] = prompt_query_join_operator.format(column_name = column_1,column_name_2 = column_2,key_1=key_1,index_1=index_1,key_2=key_2,index_2=index_2,card=card_join[join_list[i]])
        predict_list = str(sample["predicate"]).split(",")
        i = 0
        j = 1
        while i < len(predict_list) and len(predict_list) % 3 == 0:
            conditions_list = predict_list[i:i+3]
            if len(conditions_list) > 1:
                #print(conditions_list)
                column, operator, value = conditions_list
                column_table = column.split('.')[0]
                for table in column_list:
                    if column_table == table.split(" ")[-1]:
                        column_table = table
                        break
                query = f"SELECT COUNT(*) FROM {column_table} WHERE {column} {operator} {value};"
                cur_all.execute(query)
                result = cur_all.fetchone()
            input_dict["filter_{} information".format(j)] = prompt_query_predict.format(times = j,column_name=predict_list[i],operator=predict_list[i+1],value=predict_list[i+2],card = result[0])
            column_set.append(predict_list[i])
            #column_info = column_statistic_info[column_statistic_info["name"]==predict_list[i]]
            #prompt = prompt + promp_column_Cardinality.format(name=predict_list[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            j = j + 1
            i = i+3
        if len(predict_list) == 1:
            input_dict["filter information"] = "no predicate for this query"
        column_set = list(set(column_set))
        for i in range(len(column_set)):
            column_info = column_statistic_info[column_statistic_info["name"]==column_set[i]]
            #input_dict["column_{} information".format(i)] = promp_column_Cardinality.format(name=column_set[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            if pd.isna(column_info["most_common_values"].values[0]) ==False:
                if pd.isna(column_info["histogram_bounds"].values[0]) == False:
                    input_dict["column_{} information".format(i)] = promp_column_Cardinality.format(name=column_set[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0]) + promp_most.format(mcv=column_info["most_common_values"].values[0],mcf=column_info["most_common_freq"].values[0])+promp_hist.format(hist=column_info["histogram_bounds"].values[0])
                else:
                    input_dict["column_{} information".format(i)] = promp_column_Cardinality.format(name=column_set[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0]) + promp_most.format(mcv=column_info["most_common_values"].values[0],mcf=column_info["most_common_freq"].values[0])
            else:
                input_dict["column_{} information".format(i)] = promp_column_Cardinality.format(name=column_set[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
        query_sql = sample['query']
        cur_sample.execute(query_sql)
        result = cur_sample.fetchone()
        input_dict["sample information"] = prompt_samples.format(x=result[0])
        input_dict["other optimizer information"] = prompt_pg_ce.format(pg_ce = sample['pg_ce'],pr_ce=sample['pr_ce'])
        input_dict["instruction information"] = "You should give a number as the answer of all the condition: "
        return {
            "instruction": str(prompt_startword),
            "input": str(input_dict),
            "output": str(sample["Cardinal_estimation_results"]),
        }
    dataset = dataset.apply(apply_prompt_template,axis=1)
    print(dataset[0])
    '''def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["output"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))'''
    save_json = []
    for i in range(len(dataset)):
        save_json.append(dataset[i])
    json_file_path = './synthetic.json'
    json_file = open(json_file_path, mode='w')
    json.dump(save_json,json_file, indent=4)
    cur_all.close()
    conn_totaltable.close()
    cur_sample.close()
    conn_sample_info.close()
    return dataset

import re
def extract_columns_from_query(query):
    column_pattern = re.compile(r'\b(\w+\.\w+)\b')
    columns = set()
    for match in column_pattern.finditer(query):
        columns.add(match.group(1))
    
    return columns

def generate_Dataset_without_p2(splits):
    column_statistic_info = pd.read_csv('/mnt/bd/llama-finetune/llama3/llama3_test/learnedcardinalities/data/column_min_max_vals.csv',delimiter=',')
    file_path = './output_train_x_set.sql'
    with open(file_path, 'r') as f:
        lines_sql = f.readlines()
    query_workload = [lines.split('||')[0] for lines in lines_sql]
    truth = [lines.split('||')[1] for lines in lines_sql]
    pg_ce = [lines.split('||')[2] for lines in lines_sql]
    pr_ce = [lines.split('||')[3].replace('\n','') for lines in lines_sql]
    dataset = []
    for i in range(len(query_workload)):
        query = query_workload[i]
        input_json = dict()
        columns = extract_columns_from_query(query)
        input_json['query'] = query
        #print(columns)
        cnt_col = 0
        for column in columns:
            column = column.replace('imdb_t','t').replace('imdb_mii','mi_idx').replace('imdb_mi','mi').replace('imdb_ci','ci').replace('imdb_mc','mc').replace('imdb_mk','mk')
            cnt_col += 1
            column_info = column_statistic_info[column_statistic_info["name"]==column]
            #print(column,column_info)
            #input_dict["column_{} information".format(i)] = promp_column_Cardinality.format(name=column_set[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            if pd.isna(column_info["most_common_values"].values[0]) ==False:
                if pd.isna(column_info["histogram_bounds"].values[0]) == False:
                    input_json["column_{} information".format(cnt_col)] = promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0]) + promp_most.format(mcv=column_info["most_common_values"].values[0],mcf=column_info["most_common_freq"].values[0])+promp_hist.format(hist=column_info["histogram_bounds"].values[0])
                else:
                    input_json["column_{} information".format(cnt_col)] = promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0]) + promp_most.format(mcv=column_info["most_common_values"].values[0],mcf=column_info["most_common_freq"].values[0])
            else:
                input_json["column_{} information".format(cnt_col)] = promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
        input_json["other optimizer information"] = prompt_pg_ce.format(pg_ce = pg_ce[i],pr_ce=pr_ce[i])
        input_json["instruction information"] = "You should give a number as the answer of all the condition: "
        dataset.append({
            "instruction":prompt_startword,
            "input":str(input_json),
            "output":truth[i]
        })
    save_json = []
    for i in range(len(dataset)):
        save_json.append(dataset[i])
    json_file_path = './train_x_set.json'
    json_file = open(json_file_path, mode='w')
    json.dump(save_json,json_file, indent=4)
    return dataset


def generate_Dataset_with_insert_and_update(splits):
    column_statistic_info = pd.read_csv('/mnt/bd/llama-finetune/pilotscope/stats/data_info/column_min_max_values.csv',delimiter=',')
    file_path ='/mnt/bd/llama-finetune/llama3/ALECE/data/STATS/workload/upd_heavy/workload.sql'
    with open(file_path, 'r') as f:
        lines_sql = f.readlines()
    query_workload = []
    pg_ce = []
    truth = []
    for line in lines_sql:
        query_type = line.split(": ")
        if len(query_type) < 2:
            continue
        if query_type[0].count("train"):
            query_workload.append(query_type[0].split('||')[0])
            if query_type[0].count("train_sub_query") == 0:
                pg_ce.append(query_type[1].split('||')[2])
                truth.append(query_type[1].split('||')[3])
            else:
                pg_ce.append(query_type[1].split('||')[1])
                truth.append(query_type[1].split('||')[2])
    #pr_ce = [lines.split('||')[2] for lines in lines_r]
    with open('/mnt/bd/llama-finetune/llama3/ALECE/exp/STATS/e2e/ALECE_STATS_upd_upd.txt','r') as file:
        pr_ce = file.readlines()
    pr_ce = [round(float(res.replace('\n','')),2) for res in pr_ce]
    print(len(pr_ce),len(truth))
    dataset = []
    for i in range(len(query_workload)):
        query = query_workload[i]
        input_json = dict()
        columns = extract_columns_from_query(query)
        input_json['query'] = query
        #print(columns)
        cnt_col = 0
        for column in columns:
            column = column.replace('imdb_t','t').replace('imdb_mii','mi_idx').replace('imdb_mi','mi').replace('imdb_ci','ci').replace('imdb_mc','mc').replace('imdb_mk','mk')
            cnt_col += 1
            column_info = column_statistic_info[column_statistic_info["name"]==column]
            #print(column,column_info)
            #input_dict["column_{} information".format(i)] = promp_column_Cardinality.format(name=column_set[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            if pd.isna(column_info["most_common_values"].values[0]) ==False:
                if pd.isna(column_info["histogram_bounds"].values[0]) == False:
                    input_json["column_{} information".format(cnt_col)] = promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0]) + promp_most.format(mcv=column_info["most_common_values"].values[0],mcf=column_info["most_common_freq"].values[0])+promp_hist.format(hist=column_info["histogram_bounds"].values[0])
                else:
                    input_json["column_{} information".format(cnt_col)] = promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0]) + promp_most.format(mcv=column_info["most_common_values"].values[0],mcf=column_info["most_common_freq"].values[0])
            else:
                input_json["column_{} information".format(cnt_col)] = promp_column_Cardinality.format(name=column,maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
        input_json["other optimizer information"] = prompt_pg_ce.format(pg_ce = pg_ce[i],pr_ce=pr_ce[i])
        input_json["instruction information"] = "You should give a number as the answer of all the condition: "
        dataset.append({
            "instruction":prompt_startword,
            "input":str(input_json),
            "output":truth[i]
        })
    save_json = []
    for i in range(len(dataset)):
        save_json.append(dataset[i])
    json_file_path = './upd_x_set.json'
    json_file = open(json_file_path, mode='w')
    json.dump(save_json,json_file, indent=4)
    return dataset

generate_Dataset_with_insert_and_update('train')
#python -m llama_recipes.finetuning        --use_peft --peft_method lora --quantization --use_fp16 --model_name /home/liuliangzu/llama2-hf/ --dataset custom_dataset --custom_dataset.file "./dataset_prepare.py:get_preprocessed_query" --output_dir /home/liuliangzu/output_query_2/ --batch_size_training 1 --num_epochs 1 --use_fast_kernels
#get_preprocessed_arithmetic(None,tokenizer=None,split="train")
#data_set_path = './data/train.csv'
'''a = pd.read_csv('./data/train.csv',delimiter='#',header = None,names=['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results','index'])
a = a.drop(columns=['index'])
print(a.head(5))
a.to_csv('./hf_set.csv',index=False)
model_path = '/mnt/bd/llama-finetune/llama3/Meta-Llama-3-70B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
get_preprocessed_query('train',tokenizer,'train')'''
#torchrun --nnodes 1 --nproc_per_node 2 /mnt/bd/llama-finetune/llama3/llama-recipes/recipes/finetuning/finetuning.py --enable_fsdp --use_peft --peft_method lora --quantization --use_fp16 --model_name /mnt/bd/llama-finetune/llama3/Meta-Llama-3-70B-Instruct --dataset custom_dataset --custom_dataset.file "./dataset_prepare.py:get_preprocessed_query" --output_dir ../lora_test/ --batch_size_training 8 --num_epochs 1 --use_fast_kernels
#python3 -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --use_fp16 --model_name /mnt/bd/llama-finetune/llama3/Meta-Llama-3-70B-Instruct --dataset custom_dataset --custom_dataset.file "./dataset_prepare.py:get_preprocessed_query" --output_dir ../lora_test/ --batch_size_training 8 --num_epochs 1 --use_fast_kernels
