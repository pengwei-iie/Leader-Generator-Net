# Leader-Generator Net: Dividing Skill and Implicitness for Conquering FairytaleQA

This repository contains the FairytaleQA dataset and NarrativeQA1.1 dataset for our paper: [```Leader-Generator Net: Dividing Skill and Implicitness for Conquering FairytaleQA.```]( ) [Accepted to SIGIR 2023]

## Data

1. origin data in BART_DATA.zip or narrative/*.txt file (*.raw file) format: story_name <SEP> texts <SEP> question <SEP> ans1 <SEP> attr <SEP> ex_im
`(*.raw file or *.txt file)`
2. cd raw_2_pkl & python process.py to get *_source.pkl file
3. put pkl file into the datasets/data/ & python process.py to get *_source file & copy *_source file to datasets/naqa/
4. cd config & vi *.json
5. cd preprocessing/ & python preproce_naqa.py to get train_data.pkl for CL (datasets/preprocessed/NAQA/)
6. train CL model main_supcon.py
7. using the file in step 3 and model in step 6 to train BART_PG_QA for QA task


## FQA_SCL_PG DIR
### Dependency
```
# conda create -n qa python=3.8
conda activate qa
cd FQA_SCL_PG/transformers
pip install -e ".[testing]"
pip install -r ./examples/requirements.txt
pip install torchtext==0.8.0 torch==1.7.1 pytorch-lightning==1.2.2
pip install pytorch_lightning==0.9.0
pip install rouge_score
pip install sacrebleu
pip install numpy==1.23.4 (if raise AttributeError)
cd ./..
```

### FQA_SCL_PG
- get EXIM data
```
cd data
python process.py exim all fairyqa
or
python process.py exim all naqa
cd ../
```
- put data (./data/fairyqa or ./data/naqa) to **FQA_SCL_PG**/Fine-tune_BART/data/
- put bert checkpoint (./bert_128_fairyqa or ./bert_128_naqa) to **FQA_SCL_PG**/Fine-tune_BART/save/
- train (modify: data_dir, output_dir, **scl_path**, **pgnet**)
```
# cl
python 1_train.py --data_dir=./data/fairyqa/exim/ --model_name_or_path=facebook/bart-large --tokenizer_name=facebook/bart-large --config_name=facebook/bart-large --do_train --max_epochs=20 --learning_rate=5e-6 --train_batch_size=1 --max_target_length=900 --val_max_target_length=900 --test_max_target_length=900 --eval_batch_size=1 --cache_dir=./pretrained/ --output_dir=./results/fairyqa_scl_1024_128_pg_exim --gpus=1 --scl_path=./save/bert_128_fairyqa/cl/last.pth --pgnet exim
# non_cl
python 1_train.py --data_dir=./data/fairyqa/exim/ --model_name_or_path=facebook/bart-large --tokenizer_name=facebook/bart-large --config_name=facebook/bart-large --do_train --max_epochs=20 --learning_rate=5e-6 --train_batch_size=1 --max_target_length=900 --val_max_target_length=900 --test_max_target_length=900 --eval_batch_size=1 --cache_dir=./pretrained/ --output_dir=./results/fairyqa_scl_1024_no128_pg_exim --gpus=1 --scl_path=./save/bert_128_fairyqa/non_cl/last.pth --pgnet exim
```
- test (modify: data_dir, model_path, output_name)
```
# cl
python QA_predict.py --data_dir=data/fairyqa/exim/ --model_name_or_path=facebook/bart-large --tokenizer_name=facebook/bart-large --config_name=facebook/bart-large --scl_path=save/bert_128_fairyqa/cl/last.pth --model_path=results/fairyqa_scl_1024_128_pg_exim/epoch=0.ckpt --output_name=fairyqa_scl_1024_128_pg_exim/e0 --pgnet exim
# non_cl
python QA_predict.py --data_dir=data/fairyqa/exim/ --model_name_or_path=facebook/bart-large --tokenizer_name=facebook/bart-large --config_name=facebook/bart-large --scl_path=save/bert_128_fairyqa/non_cl/last.pth --model_path=results/fairyqa_scl_1024_no128_pg_exim/epoch=0.ckpt --output_name=fairyqa_scl_1024_no128_pg_exim/e0 --pgnet exim
```

## Citation
Our Paper is accepted to SIGIR 2023, you may cite:
```
WAITING
```
