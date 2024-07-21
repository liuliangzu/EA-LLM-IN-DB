import copy
import datasets
import pandas as pd
# example data preprocess

column_statistic_info = pd.read_csv('/home/liuliangzu/learnedcardinalities-master/data/column_min_max_vals.csv')

prompt_template = (
        "You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )

prompt_startword = (
    "You are a DBMS, you should use query to finish the Cardinal_estimation task, now the information will be listed: \n"
)
prompt_table = (
    "the number {times} table name : {table_name} \n"
)
prompt_query_predict = (
    "the number {times} query predict will be listed: \n"
    "----predict column is : {column_name} \n"
    "----predict operator is : {operator} \n"
    "----predict value is : {value} \n"
)
prompt_query_join_operator = (
    "the query join operator is: {join_operator} \n"
)
promp_column_Cardinality = (
    "the column {name} information max_val = {maxv}, min_val = {minv}, cardinality = {card}, num_unique_values = {unique} \n "
)
prompt_answer = (
    "---Answer is: \n"
)
'''
prompt_index = (
    "the index column : {index_column_name}\n"
    "----index type : {index_type}\n"
    "----index column max value: {index_max}\n"
    "----index column min value: {index_min}\n"
    "----index column cardinality : {cardinality}\n"
    "----index column unique nums : {num_unique_values}\n"
)'''

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
        data_files={splits: '/home/liuliangzu/learnedcardinalities-master/hf_set.csv'}
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
    def apply_prompt_template(sample):
        prompt = prompt_startword
        column_list = str(sample["TableName_or_ColumnName"]).split(",")
        for i in range(len(column_list)):
            column_name = column_list[i].split(" ")[1]
            prompt = prompt + prompt_table.format(times=i, table_name=column_name)
        prompt = prompt + prompt_query_join_operator.format(join_operator = sample["join_operater"])
        predict_list = str(sample["predicate"]).split(",")
        i = 0
        while i < len(predict_list) and len(predict_list) % 3 == 0:
            prompt = prompt + prompt_query_predict.format(times = i,column_name=predict_list[i],operator=predict_list[i+1],value=predict_list[i+2])
            column_info = column_statistic_info[column_statistic_info["name"]==predict_list[i]]
            prompt = prompt + promp_column_Cardinality.format(name=predict_list[i],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            i = i+3
        if len(predict_list) == 1:
            prompt = prompt + "no predicate for this query \n"
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
    
#python -m llama_recipes.finetuning        --use_peft --peft_method lora --quantization --use_fp16 --model_name /home/liuliangzu/llama2-hf/ --dataset custom_dataset --custom_dataset.file "./dataset_prepare.py:get_preprocessed_query" --output_dir /home/liuliangzu/output_query_2/ --batch_size_training 1 --num_epochs 1 --use_fast_kernels
#get_preprocessed_arithmetic(None,tokenizer=None,split="train")
#data_set_path = './data/train.csv'
'''a = pd.read_csv(data_set_path,delimiter='#',header = None,names=['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results'])
print(a.head(5))
a.to_csv('./hf_set.csv',index=False)'''
#get_preprocessed_query('train',None,'train')
