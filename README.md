# GLMs for Cardinality Estimation 
## Data
Here we have provided the relevant datasets for prefinetune, finetune, and test.
### Prefinetune
The prefinetune dataset used in our experiments is also updated in the /workload/prefinetune path, and the source of the dataset is explained in detail in the paper
### Finetune
The dataset sql is listed at /workload/finetune/ , we provide imdb,stats,genome,ergastf1 workloads.sql for finetune.
Each row of data has three components: the query, the database prediction, and the true result
### Test
For test sets, we provide test files for the corresponding Finetune dataset, which you can refer to to generate your own SQL tests.
### Json
As for the files that need to be used in GLMS are generated in JSON format, we provide two files, json_generate.py and json_process.py, which are used to generate various custom JSON and process various custom JSON, respectively.
### Database
We provide databse_download scripts to download the corresponding database and install it in PostgreSQL, and then you can access PostgreSQL to get various table structure information and the required prompt information, including the estimated results of PostgreSQL
## Train
For the fine-tuning of GLMS, we use [LLAMA-Factory](https://github.com/hiyouga/LLaMA-Factory/), an excellent open-source training framework, and the specific training method can refer to the corresponding framework
## Evaluate
### api_test
For specific testing, we use GLMS local deployment, and then use API calls to test the test, which can be tested by referring to the script we provide
### End to End
We also provide end-to-end test scripts, so you only need to provide the complete query collection and the corresponding estimates.
You'll need to install [Pilotscope](https://github.com/alibaba/pilotscope/)
