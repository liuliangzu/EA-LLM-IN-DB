from openai import OpenAI
import json
import pandas as pd
client = OpenAI(base_url="http://localhost:8000/v1",api_key="EMPTY")
with open('./llama3_test/synthetic_prc.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
results_list = []
for i in range(len(data)):# ,just give a number as the answer and don't say anything else
    response = client.chat.completions.create(
        model='deepseek',
        messages=[
            {"role": "system", "content": str(data[i]['instruction'])},
            {"role": "user", "content": data[i]['input']},
        ],
        max_tokens=20,
        temperature=0.7,
        stream=False
    )
    
    results_list.append([response.choices[0].message.content,data[i]['output']])
    print(results_list[i])
result_dataframe = pd.DataFrame(results_list,columns=["preds","true"])
result_dataframe.to_csv('./api_synthetic_prc_deepseek.csv',index=False)
'''
from openai import OpenAI
import json
import pandas as pd
client = OpenAI(api_key="sk-ef817b356c5b428c913d990c08561e7c", base_url="https://api.deepseek.com")
with open('./llama3_test/synthetic_eval.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

results_list = []
for i in range(len(data)):# ,just give a number as the answer and don't say anything else
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": str(data[i]['instruction'])},
            {"role": "user", "content": data[i]['input']},
        ],
        max_tokens=20,
        temperature=0.1,
        stream=False
    )

    results_list.append([response.choices[0].message.content,data[i]['output']])
result_dataframe = pd.DataFrame(results_list,columns=["preds","true"])
result_dataframe.to_csv('./api_synthetic_res.csv',index=False)'''
