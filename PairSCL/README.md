## prepare
pip install tqdm
pip install transformers
sudo apt-get install python3-tk 


## SNLI + BERT
> python preprocessing/fetch_data.py
> python preprocessing/preprocess_snli.py

> python main_supcon.py --dataset SNLI  --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32

> python main_validate.py --dataset SNLI --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32 --ckpt save/SNLI_models/SNLI_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --label_num 3

> python main_test.py --dataset SNLI --batch_size 256 --max_seq_length 128 --workers 16 --ckpt_bert save/SNLI_models/SNLI_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --ckpt_classifier save/SNLI_models/SNLI_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/classifier_last.pth --label_num 3

## SKILL 数据准备
> cd datasets/data
> python process.py
> mv skill/ ../
> cd ../

是把pkl变成txt

## SKILL + BERT
> python preprocessing/preprocess_skill.py
input: datasets/skill/*_source file
output: pkl file

> python main_supcon.py --dataset SKILL  --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32
input: pkl文件

> python main_validate.py --dataset SKILL --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32 --ckpt save/SKILL_models/SKILL_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --label_num 7

> python main_test.py --dataset SKILL --batch_size 256 --max_seq_length 128 --workers 16 --ckpt_bert save/SKILL_models/SKILL_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --ckpt_classifier save/SKILL_models/SKILL_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/classifier_last.pth --label_num 7


## SKILL + BART
> python preprocessing/preprocess_skill.py

> python bart_train.py --dataset SKILL  --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32

> python bart_validate.py --dataset SKILL --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32 --ckpt save/SKILL_models/SKILL_BART_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --label_num 7

> python bart_test.py --dataset SKILL --batch_size 256 --max_seq_length 128 --workers 16 --ckpt_bart save/SKILL_models/SKILL_BART_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --ckpt_classifier save/SKILL_models/SKILL_BART_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/classifier_last.pth --label_num 7


## SKILL_Question + BERT
> python preprocessing/preprocess_skill_Q.py

> python main_supcon.py --dataset SKILL_Q  --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32

> python main_validate.py --dataset SKILL_Q --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32 --ckpt save/SKILL_Q_models/SKILL_Q_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --label_num 7

> python main_test.py --dataset SKILL_Q --batch_size 256 --max_seq_length 128 --workers 16 --ckpt_bert save/SKILL_Q_models/SKILL_Q_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --ckpt_classifier save/SKILL_Q_models/SKILL_Q_BERT_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/classifier_last.pth --label_num 7


## SKILL_Question + BART
> python preprocessing/preprocess_skill_Q.py

>python bart_train.py --dataset SKILL_Q  --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32

> python bart_validate.py --dataset SKILL_Q --batch_size 256 --max_seq_length 128 --epochs 10 --learning_rate 5e-05 --workers 32 --ckpt save/SKILL_Q_models/SKILL_Q_BART_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --label_num 7

> python bart_test.py --dataset SKILL_Q --batch_size 256 --max_seq_length 128 --workers 16 --ckpt_bart save/SKILL_Q_models/SKILL_Q_BART_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/last.pth --ckpt_classifier save/SKILL_Q_models/SKILL_Q_BART_lr_5e-05_decay_0.0001_bsz_256_temp_0.05/classifier_last.pth --label_num 7
