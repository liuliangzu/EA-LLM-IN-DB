#from model import MLP
import torch
import numpy as np
import pandas as pd
import sys
import os
from peft import LoraConfig, get_peft_model,PeftModel

from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer,TextStreamer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
column_statistic_info = pd.read_csv('./data/column_min_max_vals.csv')

prompt_template = (
        "You are a DBMS, you should use query's information to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )

prompt_startword = (
    "You are a DBMS, you should use query's information to finish the Cardinal_estimation task, now the information will be listed: \n"
)
prompt_table = (
    "the number {times} table name : {table_name} \n"
)
prompt_query_predict = (
    "the number {times} predict will be listed: \n"
    "----predict column is : {column_name} \n"
    "----predict operator is : {operator} \n"
    "----predict value is : {value} \n"
)
prompt_query_join_operator = (
    "the join operator is: {join_operator} \n"
)
promp_column_Cardinality = (
    "the column {name} information max_val = {maxv}, min_val = {minv}, cardinality = {card}, num_unique_values = {unique} \n "
)
prompt_answer = (
    "---just give the final answer of the total query as a number and don't say anything else: \n"
)

hard_column_name = 't.production_year'
most_common_vals = [2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2013,2001,2000,1999,1998,1997,1996,1995,1994,1992,1993,1990,1991,1987,1989,1985,1988,1986,1966,1982,1978,1973,1980,1981,1965,1970,1976,1983,1979,1967,1971,1984,1975,1969,1977,1968,1974,1972,1961,1963,1962,1964,1959,1960,1955,1957,1958,1915,1913,1914,1953,1956,1911,1954,1912,1952,1951,1916,1909,1949,1917,1910,1920,1922,1939,1919,1918,1950,2014,1921,1935,1924,1930,1903,1938,1940,1926,1928,1933,1941,1932,1925,1929,1934,1948,1927,1944,1931,1937]
most_common_freqs = [0.06656667,0.06273333,0.056633335,0.050066665,0.049133334,0.048433334,0.0434,0.0372,0.031733334,0.026666667,0.025933333,0.025633333,0.0236,0.021933334,0.0209,0.018633334,0.015733333,0.015266667,0.0137,0.012366666,0.010233333,0.0097,0.009266667,0.009133333,0.008266667,0.0079666665,0.0077333334,0.0075,0.0066333334,0.0063,0.006,0.005933333,0.0059,0.0059,0.0058333334,0.0057,0.0056666667,0.0056666667,0.0056666667,0.005566667,0.0055333334,0.0055,0.0054666665,0.0053666667,0.0053,0.005266667,0.005133333,0.005,0.004833333,0.0047333334,0.0047333334,0.0046333335,0.0044,0.0039333333,0.0038666667,0.0034,0.0034,0.0033666666,0.0033,0.0031666667,0.0031666667,0.0031666667,0.0031,0.0030333332,0.0030333332,0.0029666666,0.0026666666,0.0024666667,0.0020333333,0.0019666667,0.0018333333,0.0018,0.0017,0.0015666666,0.0014333334,0.0014333334,0.0013666666,0.0013333333,0.0013333333,0.0012666667,0.0012333334,0.0012,0.0011666666,0.0011666666,0.0011333333,0.0011333333,0.0011,0.0010666667,0.0010666667,0.0010666667,0.0010666667,0.0010333334,0.0009666667,0.0009666667,0.0009666667,0.0009666667,0.00093333336,0.00093333336,0.0009,0.0009]

prompt_hard_column = (
    "there is a hard column in this query names = {}\n"
    "----the column most common vals = {}\n"
    "----the column most common freqs = {}\n"
).format(hard_column_name,most_common_vals,most_common_freqs)

def workload_proprecess(load_path):
    data_set_path = load_path
    a = pd.read_csv(data_set_path,delimiter='#',header = None,names=['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results','index'])
    print(a.head(5))
    return a

def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)

def evaluation_mscn(model_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('resnet.pth')
    model.to(device)
    pass

def evaluation_llm(lora_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/liuliangzu/llama2-hf/'
    model_llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    model = PeftModel.from_pretrained(model_llm, lora_path)
    model.to(device)
    prompt_template = (
        "You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )
    avg_acc_loss = 0
    label = data["Cardinal_estimation_results"]
    res = list()
    for i in range(len(data)):
        prompt = prompt_template.format(
                table=data["TableName_or_ColumnName"][i],
                operater=data["join_operater"][i],
                predicate=data["predicate"][i]
            )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_gen_len = 9
        prompt_len = input_ids.shape[-1]     
        generated_ids = model.generate(input_ids, max_length=prompt_len+max_gen_len, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        last_line = int(generated_text.splitlines()[-1].replace(" ",""))
        #print(last_line,label[i])
        res.append(last_line)
        loss_i = abs(last_line - label[i])
        avg_acc_loss = avg_acc_loss + loss_i
    print("llama2 total loss: {}\n".format(avg_acc_loss))
    return res

def evaluation_llm_stinfo(lora_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/liuliangzu/llama2-hf/'
    model_llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    model = PeftModel.from_pretrained(model_llm, lora_path)
    model.to(device)
    avg_acc_loss = 0
    label = data["Cardinal_estimation_results"]
    res = list()
    for i in range(len(data)):
        sample_column = data["TableName_or_ColumnName"][i]
        sample_join = data["join_operater"][i]
        sample_predict = data["predicate"][i]
        prompt = prompt_startword
        column_list = str(sample_column).split(",")
        for j in range(len(column_list)):
            column_name = column_list[j].split(" ")[1]
            prompt = prompt + prompt_table.format(times=j, table_name=column_name)
        prompt = prompt + prompt_query_join_operator.format(join_operator = sample_join)
        predict_list = str(sample_predict).split(",")
        '''if hard_column_name in predict_list:
            prompt = prompt + prompt_hard_column'''
        j = 0
        cnt = 1
        while j < len(predict_list) and len(predict_list) % 3 == 0:
            cnt = cnt + 1
            prompt = prompt + prompt_query_predict.format(times = cnt,column_name=predict_list[j],operator=predict_list[j+1],value=predict_list[j+2])
            column_info = column_statistic_info[column_statistic_info["name"]==predict_list[j]]
            prompt = prompt + promp_column_Cardinality.format(name=predict_list[j],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            j = j+3
        if len(predict_list) == 1:
            prompt = prompt + "no predicate for this query \n"
        prompt = prompt + prompt_answer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_gen_len = 9
        prompt_len = input_ids.shape[-1]     
        generated_ids = model.generate(input_ids, max_length=prompt_len+max_gen_len, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        last_line = int(generated_text.splitlines()[-1].replace(" ",""))
        #print(generated_text,last_line,label[i])
        res.append(last_line)
        loss_i = abs(last_line - label[i])
        avg_acc_loss = avg_acc_loss + loss_i
    print("llama2 total loss: {}\n".format(avg_acc_loss))
    return res

def evaluation_phi(lora_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/liuliangzu/phi_11_epoch/'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #model = PeftModel.from_pretrained(model_llm, lora_path)
    model.to(device)
    avg_acc_loss = 0
    label = data["Cardinal_estimation_results"]
    res = list()
    for i in range(len(data)):
        sample_column = data["TableName_or_ColumnName"][i]
        sample_join = data["join_operater"][i]
        sample_predict = data["predicate"][i]
        prompt = prompt_startword
        column_list = str(sample_column).split(",")
        for j in range(len(column_list)):
            column_name = column_list[j].split(" ")[1]
            prompt = prompt + prompt_table.format(times=j, table_name=column_name)
        prompt = prompt + prompt_query_join_operator.format(join_operator = sample_join)
        predict_list = str(sample_predict).split(",")
        j = 0
        cnt = 1
        while j < len(predict_list) and len(predict_list) % 3 == 0:
            cnt = cnt + 1
            prompt = prompt + prompt_query_predict.format(times = cnt,column_name=predict_list[j],operator=predict_list[j+1],value=predict_list[j+2])
            column_info = column_statistic_info[column_statistic_info["name"]==predict_list[j]]
            prompt = prompt + promp_column_Cardinality.format(name=predict_list[j],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            j = j+3
        if len(predict_list) == 1:
            prompt = prompt + "no predicate for this query \n"
        prompt = prompt + prompt_answer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_gen_len = 1000
        prompt_len = input_ids.shape[-1]     
        generated_ids = model.generate(input_ids, max_length=prompt_len+max_gen_len, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        '''prompt_len = input_ids.shape[-1]     
        generated_ids = model.generate(input_ids, max_length=prompt_len+max_gen_len, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)'''
        print(generated_text)
        last_line = int(generated_text.splitlines()[-1])
        print(last_line,label[i])
        res.append(last_line)
        loss_i = abs(last_line - label[i])
        avg_acc_loss = avg_acc_loss + loss_i
    print("phi total loss: {}\n".format(avg_acc_loss))
    return res

def evaluation_llm_mlp(lora_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mlp = torch.load('mlp.pth')
    model_mlp.to(device)
    model_path = '/home/liuliangzu/llama2-hf/'
    model_base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    model = PeftModel.from_pretrained(model_base, lora_path)
    model.to(device)
    prompt_template = (
        "You are a DBMS, you should use query to finish the Cardinal_estimation task, the table and column is:\n"
        "{table}\n---the join_operater is:\n{operater}\n---the predicate is:\n{predicate}\n---\nAnswer:\n"
    )
    avg_acc_loss = 0
    label = data["Cardinal_estimation_results"]
    labels_norm, min_val, max_val = normalize_labels(label,None,None)
    res = list()
    for i in range(len(data)):
        prompt = prompt_template.format(
                table=data["TableName_or_ColumnName"][i],
                operater=data["join_operater"][i],
                predicate=data["predicate"][i]
            )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
        # 前向传播
        outputs = model(input_ids, output_hidden_states=True)            
        hidden_states = outputs.hidden_states[-1][:, 0, :].to(torch.float32)  # 获取最后一层隐藏状态的 [CLS] token
        # 通过 MLP
        y_pred = model_mlp(hidden_states)
        y_pred = y_pred.cpu().detach().numpy()
        label_out = unnormalize_labels(y_pred,min_val,max_val)[0][0]
        res.append(label_out)
        loss_i = abs(label_out - label[i])
        #print(y_pred,labels_norm[i],label_out,label[i],loss_i)
        avg_acc_loss = avg_acc_loss + loss_i
    print("llm_mlp total loss = {}".format(avg_acc_loss))
    return res


def evaluation_llm3_stinfo(lora_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/mnt/bd/llama-finetune/llama3/Meta-Llama-3-70B-Instruct/'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    #model.to(device)
    avg_acc_loss = 0
    label = data["Cardinal_estimation_results"]
    res = list()
    for i in range(len(data)):
        total_line = 1
        sample_column = data["TableName_or_ColumnName"][i]
        sample_join = data["join_operater"][i]
        sample_predict = data["predicate"][i]
        prompt = prompt_startword
        column_list = str(sample_column).split(",")
        for j in range(len(column_list)):
            column_name = column_list[j].split(" ")[1]
            prompt = prompt + prompt_table.format(times=j, table_name=column_name)
            total_line += 1
        prompt = prompt + prompt_query_join_operator.format(join_operator = sample_join)
        total_line += 1
        predict_list = str(sample_predict).split(",")
        '''if hard_column_name in predict_list:
            prompt = prompt + prompt_hard_column'''
        j = 0
        cnt = 1
        while j < len(predict_list) and len(predict_list) % 3 == 0:
            cnt = cnt + 1
            prompt = prompt + prompt_query_predict.format(times = cnt,column_name=predict_list[j],operator=predict_list[j+1],value=predict_list[j+2])
            column_info = column_statistic_info[column_statistic_info["name"]==predict_list[j]]
            prompt = prompt + promp_column_Cardinality.format(name=predict_list[j],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            j = j+3
            total_line += 5
        if len(predict_list) == 1:
            prompt = prompt + "no predicate for this query \n"
            total_line += 1
        prompt = prompt + prompt_answer
        total_line += 1
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_gen_len = 10
        prompt_len = input_ids.shape[-1]     
        generated_ids = model.generate(input_ids, max_length=prompt_len+max_gen_len, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        #print(generated_text)
        last_line = generated_text.splitlines()[total_line].replace(".","").split("  ")[-1].replace(" ","")
        print(last_line)
        #last_line = int(generated_text.splitlines()[-1].replace(" ",""))
        #print(generated_text,last_line,label[i])
        res.append(last_line)
        #loss_i = abs(last_line - label[i])
        #avg_acc_loss = avg_acc_loss + loss_i
    #print("llama2 total loss: {}\n".format(avg_acc_loss))
    return res

def evaluation_deepseek_stinfo(lora_path,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/mnt/bd/llama-finetune/llama3/DeepSeek-Coder-V2-Lite-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto',trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    #model.to(device)
    avg_acc_loss = 0
    label = data["Cardinal_estimation_results"]
    res = list()
    for i in range(len(data)):
        total_line = 1
        sample_column = data["TableName_or_ColumnName"][i]
        sample_join = data["join_operater"][i]
        sample_predict = data["predicate"][i]
        prompt = prompt_startword
        column_list = str(sample_column).split(",")
        for j in range(len(column_list)):
            column_name = column_list[j].split(" ")[1]
            prompt = prompt + prompt_table.format(times=j, table_name=column_name)
            total_line += 1
        prompt = prompt + prompt_query_join_operator.format(join_operator = sample_join)
        total_line += 1
        predict_list = str(sample_predict).split(",")
        '''if hard_column_name in predict_list:
            prompt = prompt + prompt_hard_column'''
        j = 0
        cnt = 1
        while j < len(predict_list) and len(predict_list) % 3 == 0:
            cnt = cnt + 1
            prompt = prompt + prompt_query_predict.format(times = cnt,column_name=predict_list[j],operator=predict_list[j+1],value=predict_list[j+2])
            column_info = column_statistic_info[column_statistic_info["name"]==predict_list[j]]
            prompt = prompt + promp_column_Cardinality.format(name=predict_list[j],maxv=column_info["max"].values[0],minv=column_info["min"].values[0],card=column_info["cardinality"].values[0],unique=column_info["num_unique_values"].values[0])
            j = j+3
            total_line += 5
        if len(predict_list) == 1:
            prompt = prompt + "no predicate for this query \n"
            total_line += 1
        prompt = prompt + prompt_answer
        total_line += 1
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_gen_len = 10
        prompt_len = input_ids.shape[-1]     
        generated_ids = model.generate(input_ids, max_length=prompt_len+max_gen_len, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        #print(generated_text)
        last_line = generated_text.splitlines()[total_line].replace(".","").split("  ")[-1].replace(" ","")
        print(last_line)
        #last_line = int(generated_text.splitlines()[-1].replace(" ",""))
        #print(generated_text,last_line,label[i])
        res.append(last_line)
        #loss_i = abs(last_line - label[i])
        #avg_acc_loss = avg_acc_loss + loss_i
    #print("llama2 total loss: {}\n".format(avg_acc_loss))
    return res

def record_res(res,label,model_name,workload_name):
    file_name = "results/predictions_" + workload_name + "_" + model_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for i in range(len(res)):
            f.write(str(res[i]) + "," + str(label[i]) + "\n")

def main():
    l3_lora_path = '/mnt/bd/llama-finetune/llama3/sft'
    ds_lora_path = '/mnt/bd/llama-finetune/llama3/deepseek_sft'
    data = workload_proprecess('./workloads/synthetic.csv')
    res = evaluation_deepseek_stinfo(ds_lora_path,data)
    #evaluation_llm_mlp('./query_encoder/',data)
    label = data["Cardinal_estimation_results"]
    record_res(res,label,"deepseek16B","synthetic")

main()
