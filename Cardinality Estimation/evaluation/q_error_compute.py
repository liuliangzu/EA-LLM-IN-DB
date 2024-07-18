import pandas as pd
import numpy as np
import os

def record_query(name,res):
    file_name = "./results/query_" + name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    data = pd.DataFrame(data = res,index = None,columns = ['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results'])
    data.to_csv(file_name)

def print_qerror(task,data_name,preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if str(preds_unnorm[i]).isdigit() and float(preds_unnorm[i])>0:
            if float(preds_unnorm[i]) > float(labels_unnorm[i]):
                qerror.append(float(preds_unnorm[i]) / float(labels_unnorm[i]))
            else:
                qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
    print("*"*50)
    print(task)
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    print("*"*50)
    '''query = []
    data_file = "./workloads/{}.csv".format(data_name)
    data = pd.read_csv(data_file,delimiter='#',header = None,names=['TableName_or_ColumnName','join_operater','predicate','Cardinal_estimation_results'])
    for i in range(len(preds_unnorm)):
        qerror_now = 0
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror_now = float(preds_unnorm[i]) / float(labels_unnorm[i])
        else:
            qerror_now = float(labels_unnorm[i]) / float(preds_unnorm[i])
        if qerror_now >= np.percentile(qerror, 90):
            query.append(data.loc[i])
    record_query(task,query)'''
    
job_mscn_llm_stinfo = pd.read_csv('/mnt/bd/llama-finetune/llama3/api_onetable_deepseek.csv',header = None,names=['preds_unnorm','labels_unnorm'])
scale_llm_10_stinfo = pd.read_csv('/mnt/bd/llama-finetune/llama3/api_scale_llama3.csv',header = None,names=['preds_unnorm','labels_unnorm'])
synthetic_llm_10_stinfo = pd.read_csv('/mnt/bd/llama-finetune/llama3/api_synthetic_llama3.csv',header = None,names=['preds_unnorm','labels_unnorm'])
print_qerror(str("job_mscn_llm_stinfo"),"scale",job_mscn_llm_stinfo['preds_unnorm'],job_mscn_llm_stinfo['labels_unnorm'])
print_qerror(str("scale_llm_10_stinfo"),"synthetic",scale_llm_10_stinfo['preds_unnorm'],scale_llm_10_stinfo['labels_unnorm'])
print_qerror(str("synthetic_llm_10_stinfo"),"synthetic",synthetic_llm_10_stinfo['preds_unnorm'],synthetic_llm_10_stinfo['labels_unnorm'])
