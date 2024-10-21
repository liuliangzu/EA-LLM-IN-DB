from openai import OpenAI
import json
import pandas as pd
import re
imdb = "other optimizer prediction"
def get_ndv_predict():
    res = []
    with open('/mnt/bd/llama-finetune/pilotscope/ndv_job_light.txt','r') as file:
        lines_sql = file.readlines()
    for line in lines_sql:
        s = line.replace('\n','').split(',')[-1]
        numbers = re.findall(r'\d+', s)
        res.append(int(numbers[0]))
    return res
client = OpenAI(base_url="http://localhost:8000/v1",api_key="EMPTY")
with open('/mnt/bd/llama-finetune/pilotscope/stats-icl.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
with open('/mnt/bd/llama-finetune/llama3/PRICE/genome_test.txt','r') as file:
    lines_sql = file.readlines()
pr_res = imdb
#pr_res = get_ndv_predict()
results_list = []
chat = 0
for i in range(len(data)):# ,just give a number as the answer and don't say anything else
#for i in range(3):# ,just give a number as the answer and don't say anything else
    q_error = 100
    other_prompt = ""
    message_history = [{"role": "system", "content": str(data[i]['instruction'])},{"role": "user", "content": data[i]['input']}]
    res = 0
    flag = 0
    max_chat = 0
    if chat == 1:
        while q_error > 2 and max_chat < 5:
            if len(other_prompt) > 5:
                flag = 1
                message_history.append(
                        {"role": "user", "content": other_prompt}
                    )
            response = client.chat.completions.create(
                model='llama',
                messages=message_history,
                max_tokens=15,
                temperature=0.7,
                stream=False
            )
            max_chat = max_chat + 1
            llm_res = response.choices[0].message.content
            print(llm_res)
            if llm_res[0] == " ":
                llm_res = llm_res.split(" ")[1]
            if float(pr_res[i]) < 1:
                pr_res[i] = 1
            else:
                pr_res[i] = int(float(pr_res[i]))
            message_history.append({"role": "assistant", "content": llm_res})
            #pr_res[i] = max(float(data[i]['output']) / 5.0,1.0)
            try:
                t = float(llm_res.split(" ")[0].split("\n")[0].split(",")[0])
            except:
                break
            if float(pr_res[i]) > float(llm_res.split(" ")[0].split("\n")[0].split(",")[0]):
                if float(llm_res.split(" ")[0].split("\n")[0].split(",")[0]) == 0:
                    q_error = float(pr_res[i])/1
                else:  
                    q_error = float(pr_res[i])/float(llm_res.split(" ")[0].split("\n")[0].split(",")[0])
            else:
                q_error = float(llm_res.split(" ")[0].split("\n")[0].split(",")[0])/float(pr_res[i])
            #other_prompt = "The other optimizier estimate this query with {}, and you should give your estimation again as a number".format(pr_res[i])
            other_prompt = "The other optimizier estimate this query with {}, then you should give your estimation again as a number".format(pr_res[i])
            if flag == 1:
                res = (float(llm_res.split(" ")[0].split("\n")[0].split(",")[0]) + float(pr_res[i]))/2
            else:
                res = float(llm_res.split(" ")[0].split("\n")[0].split(",")[0])
            #res = (float(llm_res.split(" ")[0].split("\n")[0].split(",")[0])+ float(pr_res[i]))/2
        results_list.append([res,data[i]['output']])
        print(float(pr_res[i]),res,data[i]['output'])
    else:
        response = client.chat.completions.create(
                model='deepseek',
                messages=message_history,
                max_tokens=15,
                temperature=0.7,
                stream=False
            )
        llm_res = response.choices[0].message.content
        results_list.append([llm_res,data[i]['output']])
        print(llm_res,data[i]['output'])
result_dataframe = pd.DataFrame(results_list,columns=["preds","true"])
result_dataframe.to_csv('./api_icl_stats_y_llama3.csv',index=False)
