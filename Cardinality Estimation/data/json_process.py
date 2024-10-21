import json
import pandas as pd
def filter_cut(data, filter_info):
    res = []
    for s in data:
        string = s['input']
        if s.count(filter_info):
            res.append(s)
    return res

def drop_info(data,info):
    res = []
    for s in data:
        string = s['input']
        valid_json_string = string.replace("'", '"')
        valid_json_string = json.loads(valid_json_string)
        del valid_json_string[info]
        res.append({
            "instruction":s['instruction'],
            "input":str(valid_json_string),
            "output":s['output']
        })
    return res

def in_context_insert(data):
    res = []
    cnt = 0
    for s in data:
        string = s['input']
        valid_json_string = string.replace("'", '"')
        valid_json_string = json.loads(valid_json_string)
        with open('/mnt/bd/llama-finetune/pilotscope/query_retreival_stats_test.sql','r') as file:
            lines_sql = file.readlines()
        lines_sql = lines_sql[cnt:cnt+10]
        cnt = cnt + 1
        example_context = ""
        for line in lines_sql:
            cot = line.replace("\n", '').split("||")
            str_cot = "for the example {},postgresql give a results {}, deeplearning model give a results {} and the truth is {}. ".format(cot[0],cot[2],cot[3],cot[1])
            example_context = example_context + str_cot
        valid_json_string["example_info"] = example_context
        instrution = valid_json_string['instruction information']
        del valid_json_string['instruction information']
        valid_json_string['instruction information'] = instrution
        res.append({
            "instruction":s['instruction'],
            "input":str(valid_json_string),
            "output":s['output']
        })
    return res

def column_infor_insert(data):
    res = []
    cnt = 0
    for s in data:
        string = s['input']
        string = string + "This column has 500000 rows, please give your answer as a number."
        res.append({
            "instruction":s['instruction'],
            "input":str(string),
            "output":s['output']
        })
    return res

def column_output_update(data):
    res = []
    cnt = 0
    ce = pd.read_csv('/mnt/bd/llama-finetune/pilotscope/author_name_ce.csv',names=['ce_res'])
    for i in range(len(data)):
        s = data[i]
        string = s['output']
        res.append({
            "instruction":s['instruction'],
            "input":s['input'],
            "output":str(ce['ce_res'][i])
        })
    return res

def filter_value(data):
    res = []
    cnt = 0
    #ce = pd.read_csv('/mnt/bd/llama-finetune/pilotscope/author_name_ce.csv',names=['ce_res'])
    with open('/mnt/bd/llama-finetune/llama3/lplm/title_names_training_set.txt','r') as file:
        lines_sql = file.readlines()
    with open('/mnt/bd/llama-finetune/pilotscope/title_estimated_cardinalities_train.txt','r') as file:
        lplm_ce = file.readlines()
    print(len(lines_sql),len(lplm_ce))
    for i in range(len(data)):
        s = data[i]
        string = s['output']
        string2 = s['input'].split(',')
        temp = "this query is : select count(*) from title where name LIKE '{}', ".format(lines_sql[i].replace('\n',''))
        temp2 = " and for this query the deep learning model give the estimation is {} ,".format(int(float(lplm_ce[i].replace('\n',''))))
        temp = temp + temp2
        for j in range(1,len(string2)):
            temp = temp + string2[j]
            if j != len(string2) - 1:
                temp = temp + ','
        if float(string) > 10:
            res.append({
                "instruction":s['instruction'],
                "input":temp,
                "output":s['output']
            })
    return res

def ndv_info_insert(data):
    res = []
    with open('/mnt/bd/llama-finetune/pilotscope/ndv_job_light.txt','r') as file:
        lines_sql = file.readlines()
    ground_truth = pd.read_csv('/mnt/bd/llama-finetune/pilotscope/ndv_job_truth.csv',names=['ndv','true','pg'])
    for i in range(len(data)):
        s = data[i]
        string = s['input']
        string3 = "You are a DBMS, you should use query to finish a Distinct Value Num Prediction task, target column is t.id, now the information will be listed: "
        valid_json_string = string.replace("'", '"')
        valid_json_string = json.loads(valid_json_string)
        del valid_json_string['sample information']
        del valid_json_string['other optimizer information']
        valid_json_string["other optimizer information"] = "The sample of this query is " + lines_sql[i].replace('\n','').replace('1;','{};'.format(ground_truth['ndv'][i]))
        res.append({
            "instruction":string3,
            "input":str(valid_json_string),
            "output":str(ground_truth['true'][i]).replace('[','').replace(']','')
        })
    return res
imdb = 'your optimizer prediction'
def price_insert(data):
    res = []
    cnt = 0
    with open('/mnt/bd/llama-finetune/llama3/PRICE/genome_test.txt','r') as file:
        lines_sql = file.readlines()
    for i in range(len(data)):
        s = data[i]
        string = s['input']
        valid_json_string = string.replace("'", '"')
        if valid_json_string.count('/*'):
            continue
        try:
            valid_json_string = json.loads(valid_json_string)
            #print(valid_json_string)
            str_cot = " and the deep learning model give an estimation of {} for this query".format(round(float(lines_sql[cnt].replace('\n','')),2))
            cnt = cnt + 1
            valid_json_string["other optimizer information"] = valid_json_string["other optimizer information"] + str_cot
            res.append({
                "instruction":s['instruction'],
                "input":str(valid_json_string),
                "output":s['output']
            })
        except:
            valid_json_string = valid_json_string.replace('= "', "= '")
            valid_json_string = valid_json_string.replace('")', "')")
            valid_json_string = valid_json_string.replace('" a', "' a")
            valid_json_string = valid_json_string.replace('";', "';")
            print(valid_json_string)
            valid_json_string = json.loads(valid_json_string)
            #print(valid_json_string)
            str_cot = " and the deep learning model give an estimation of {} for this query".format(lines_sql[cnt].replace('\n',''))
            cnt = cnt + 1
            valid_json_string["other optimizer information"] = valid_json_string["other optimizer information"] + str_cot
            res.append({
                "instruction":s['instruction'],
                "input":str(valid_json_string),
                "output":s['output']
            })
    return res

with open('/mnt/bd/llama-finetune/pilotscope/genome_stats_json.json', 'r') as file:
    data = json.load(file)
#res = filter_cut(data,'filter_4')
#res = drop_info(data,'sample information')
#res = column_infor_insert(data)
#res = column_output_update(res)
res = price_insert(data)
print(res[0],len(res))
json_file_path = './genome_test_.json'
json_file = open(json_file_path, mode='w')
json.dump(res,json_file, indent=4)
