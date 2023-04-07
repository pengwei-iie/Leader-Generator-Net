
import argparse
import logging
import os
from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from lightning_base import BaseTransformer   # , add_generic_args, generic_train
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import InputExample
from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from utils import (
    ROUGE_KEYS, LegacySeq2SeqDataset, Seq2SeqDataset, assert_all_frozen, calculate_bleu, calculate_rouge, flatten_list,
    freeze_params, get_git_info, label_smoothed_nll_loss, lmap, pickle_save, save_git_info, save_json,
    use_task_specific_params, parse_numeric_cl_kwargs,
)
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, self.mode)

    #         self.target_lens = {
    #             "train": self.hparams.max_target_length,
    #             "val": self.hparams.val_max_target_length,
    #             "test": self.hparams.test_max_target_length,
    #         }
    #         self.decoder_start_token_id = None  # default to config
    #         if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
    #             self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
    #             self.model.config.decoder_start_token_id = self.decoder_start_token_id
    #         self.test_max_target_length = self.hparams.test_max_target_length

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        # pad_token_id = self.tokenizer.pad_token_id
        # source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids, pred_exim = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=20,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=3,
            early_stopping=True,
            use_cache=False,
            **generate_kwargs
        )
        return generated_ids, pred_exim


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def generate_summaries_or_translations(examples: List[List], model_name: str, batch_size: int = 8,
                                       device: str = DEFAULT_DEVICE, fp16=False, task="summarization",
                                       prefix='', args=None, **generate_kwargs, ) -> Dict:
    # save outputs to a list
    output_list = []

    ## 0421 -- load model outside
    '''
    model_name = str(model_name)

    if "summarization" in args.task:
        model: SummarizationModule = SummarizationModule(args)
    else:
        model: SummarizationModule = TranslationModule(args)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()

    if args.device != 'cpu':
      model.to('cuda:{}'.format(args.device))
    '''

    # tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=args.cache_dir)
    # tokenizer.add_special_tokens({'sep_token': '<SEP>'})
    # model.resize_token_embeddings(len(tokenizer))

    # print('#############################################')
    # print("# model is loaded from", args.ckpt_path)
    # print('# tokenizer.all_special_tokens =', model.tokenizer.all_special_tokens)
    # print('# tokenizer.all_special_ids =', model.tokenizer.all_special_ids)
    # print('#############################################')
    pred_exim_lst = []
    # update config with task specific params
    use_task_specific_params(model, task)
    for k, (examples_chunk, exim_chunk) in enumerate(tqdm(zip(list(chunks(examples[0], batch_size)), list(chunks(examples[1], batch_size))))):
        examples_chunk = [prefix + text for text in examples_chunk]

        if device == 'cpu':
            batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest")
        else:
            batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(
                'cuda:{}'.format(device))

        ###batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest")

        if len(batch.input_ids[0]) > 1024:

            if device == 'cpu':
                end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:]
            else:
                end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:].to('cuda:{}'.format(device))

            ###end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:]

            input_ids = torch.cat((batch.input_ids[0][:1022], end_token), 0).unsqueeze(0)
            batch.input_ids = input_ids
        # batch.input_ids[0,-1] = 2
        # print(batch.input_ids)
        # print(batch.attention_mask)
        # return

        # bert tokenize
        input_lst, exim_lst = [], []
        for x in exim_chunk:
            exim, question, context = x.split(" <SEP> ")
            input_lst.append(InputExample(guid=str(k), text_a=question, text_b=context))
            exim_lst.append(0 if exim == "explicit" else (1 if exim == "implicit" else None))
        features = convert_examples_to_features(
            input_lst,
            model.scl_tokenizer,
            max_length=128,
            label_list=['none'],
            output_mode="classification"
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_exim_ids = torch.tensor(exim_lst, dtype=torch.long)
        generate_kwargs["scl_model"] = model.scl_model
        generate_kwargs["scl_input_ids"] = all_input_ids.to('cuda:{}'.format(device)) if device != 'cpu' else all_input_ids
        generate_kwargs["scl_attention_mask"] = all_attention_mask.to('cuda:{}'.format(device)) if device != 'cpu' else all_attention_mask
        generate_kwargs["scl_token_type_ids"] = all_token_type_ids.to('cuda:{}'.format(device)) if device != 'cpu' else all_token_type_ids
        generate_kwargs["scl_exim_ids"] = all_exim_ids.to('cuda:{}'.format(device)) if device != 'cpu' else all_exim_ids
        generate_kwargs["encoder_input_bak"] = batch.input_ids.to('cuda:{}'.format(device)) if device != 'cpu' else batch.input_ids
        generate_kwargs["using_pg_net"] = args.pgnet

        summaries, pred_exim = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            **generate_kwargs,
        )  # [1,20]
        dec = model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            output_list.append(hypothesis)
        pred_exim_lst += [pred_exim]

    return output_list, pred_exim_lst


def run_generate(examples):
    parsed = {}
    generate_results = generate_summaries_or_translations(examples, args.model_name_or_path, batch_size=args.bs,
                                                          device=args.device, fp16=args.fp16, task=args.task,
                                                          prefix=args.prefix, args=args, **parsed, )  # 生成
    return generate_results


# For QA Task
def e2eQA(source_dir, dataset_type, output_name):
    input_for_QA_model, exim_label = [[], []], []
    QC_lines = open(os.path.join(source_dir, f'{dataset_type}.source'), 'r').readlines()
    for i in QC_lines:
        exim, question, content = i.strip().split(" <SEP> ")
        exim_label.append(exim)
        input_for_QA_model[0].append(question + " <SEP> " + content)  # question <SEP> text
        input_for_QA_model[1].append(exim + " <SEP> " + question + " <SEP> " + content)
        # attr, question, passage = i.replace('\n', '').split('<SEP>')
        # input_for_QA_model.append(f'Skill: {attr} <SEP> Question: {question} <SEP> Passage: {passage}')  # Skill: xxx <SEP> Question: xxx <SEP> Passage: xxx

    output_for_QA_model, pred_exim = run_generate(input_for_QA_model)  # ans
    for i in range(len(output_for_QA_model)):
        output_for_QA_model[i] = output_for_QA_model[i].replace("\n", "").split(".")[0].strip(" .") + " ."  # 末尾为" ."，后面的都去掉

    assert len(input_for_QA_model[0]) == len(output_for_QA_model)
    os.makedirs(os.path.join(source_dir, output_name), exist_ok=True)
    f = open(os.path.join(source_dir, output_name, f'predict_{dataset_type}.txt'), 'w')
    f.write("exim" + "\t" + "ans" + "\n")
    for i in range(len(output_for_QA_model)):
        f.write(str(pred_exim[i]) + '\t' + output_for_QA_model[i] + '\n')
    f.close()

    exim_map = {"explicit": "0", "implicit": "1"}
    exim_acc = sum([str(a) == exim_map[b] for a, b in zip(pred_exim, exim_label)])/len(exim_label)
    print(f"exim acc: {exim_acc}")

    return output_for_QA_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--data_dir", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--prefix", type=str, required=False, default='',
                        help="will be added to the begininng of src examples")
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=1, required=False, help="batch size")
    parser.add_argument("--n_obs", type=int, default=-1, required=False,
                        help="How many observations. Defaults to all.")
    parser.add_argument("--fp16", action="store_true")
    ################################
    parser.add_argument("--cache_dir", default="./pretrained/", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--ckpt_path", type=str, help='path to stored model checkpoints',)
    parser.add_argument("--scl_path", type=str, help='path of PairScl bert model checkpoint')
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/bart-large",
                        help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--config_name", type=str, default="facebook/bart-large")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-large")
    parser.add_argument("--test_max_target_length", type=int)
    parser.add_argument("--eval_max_length", type=int)
    parser.add_argument("--model_path", type=str, help='path of Bart model checkpoint')
    parser.add_argument("--pgnet", default='none', type=str, choices=['none', 'ori', 'exim', 'random'], help='pgnet type')
    ################################
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()

    model: SummarizationModule = SummarizationModule(args)
    # print(model)
    MODEL_PATH = args.model_path
    # MODEL_PATH = os.path.join('save', 'epoch=4.ckpt')  # N  p_e_1.ckpt
    # MODEL_PATH = os.path.join(conf.data_dir, 'save', 'pretrained_BART_on_Fairytale_5e-6_b1_epoch=1.ckpt')  # F
    print(MODEL_PATH)
    model = model.load_from_checkpoint(MODEL_PATH, map_location='cpu')
    model.eval()
    if DEFAULT_DEVICE != 'cpu':
        model.to('cuda:{}'.format(DEFAULT_DEVICE))

    print('#############################################')
    print("# model is loaded from", MODEL_PATH)
    print('# tokenizer.all_special_tokens =', model.tokenizer.all_special_tokens)
    print('# tokenizer.all_special_ids =', model.tokenizer.all_special_ids)
    print('#############################################')

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bart_skill_dir = args.data_dir  # os.path.join('data', 'BART_QA')  # BART_SKILL
    for m in ['val', 'test']:
        output_for_test = e2eQA(bart_skill_dir, m, args.output_name)  # question <SEP> text
        res, len_f = 0, 0
        with open(os.path.join(bart_skill_dir, f'{m}_two_answer.target'), 'r', encoding='utf-8') as f:
            for row, out in zip(f, output_for_test):
                ans1, ans2 = row.strip().split(' </s> ')
                res += max(scorer.score(ans1, out)['rougeL'].fmeasure,
                           scorer.score(ans2, out)['rougeL'].fmeasure)
                len_f += 1
            print(f"The average rougeL score on {m} is: ", res/len_f)
