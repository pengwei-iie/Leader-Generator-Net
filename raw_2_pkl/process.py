#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number

"""
Author: Wesley (liwanshui12138@gmail.com)
Date: 2022-5-7
"""
import pickle
import sys
import os

mode = sys.argv[1]  # qa, skill, cl, debug, exim 就是生成
split_ = sys.argv[2]  # train, val, test, all
data_src = sys.argv[3]  # fairyqa, naqa


split_type_lst = [split_] if split_ in ['train', 'val', 'test'] else ['train', 'val', 'test']

for split_type in split_type_lst:
    print('split_type: ', split_type)
    qc_dic, ans_dic = {}, {}
    story_set, attr_set = [], []

    with open(f'{data_src}/{split_type}_{data_src}.raw', 'r', encoding='utf-8') as f:
        for row in f:
            story_name, text, question, ans, attr, ex_im = row.strip().split(' <SEP> ')
            if story_name not in story_set:
                story_set.append(story_name)
            if attr not in attr_set:
                attr_set.append(attr)

            key1, key2 = f'{story_name}_{attr}', f'{story_name}_{ex_im}'

            # qc
            val_bug_1 = f'{story_name} <SEP> {attr} <SEP> {question} <SEP> {text}'
            val_bug_2 = f'{story_name} <SEP> {ex_im} <SEP> {question} <SEP> {text}'
            val_skill = f'Skill: {attr} <SEP> Question: {question} <SEP> Passage: {text}'
            val_cl = f'{question} <SEP> {text}'
            if key1 not in qc_dic: qc_dic[key1] = []
            if key2 not in qc_dic: qc_dic[key2] = []
            val_1 = val_bug_1 if mode == 'debug' else (val_skill if mode == 'skill' else val_cl)
            val_2 = val_bug_2 if mode == 'debug' else (val_skill if mode == 'skill' else val_cl)
            qc_dic[key1].append(val_1)
            qc_dic[key2].append(val_2)

            # ans
            val_bug_1 = f'{story_name} <SEP> {attr} <SEP> {ans}'
            val_bug_2 = f'{story_name} <SEP> {ex_im} <SEP> {ans}'
            val_skill = ans
            val_cl = ans
            if key1 not in ans_dic: ans_dic[key1] = []
            if key2 not in ans_dic: ans_dic[key2] = []
            val_1 = val_bug_1 if mode == 'debug' else (val_skill if mode == 'skill' else val_cl)
            val_2 = val_bug_2 if mode == 'debug' else (val_skill if mode == 'skill' else val_cl)
            ans_dic[key1].append(val_1)
            ans_dic[key2].append(val_2)

    print(f'story num: {len(story_set)}, attr num: {len(attr_set)}')
    print('d')

    sort_story_set, sort_attr_set = sorted(story_set), sorted(attr_set)
    sort_attr_set = sort_attr_set + ['explicit', 'implicit']
    if not os.path.exists(f'{data_src}/name'):
        os.mkdir(f'{data_src}/name')
    with open(f"{data_src}/name/{split_type}_story_name.txt", 'w', encoding='utf-8') as f1:
        for i in sort_story_set:
            f1.write(i + '\n')
    with open(f"{data_src}/name/attribute_exim_name.txt", 'w', encoding='utf-8') as f2:
        for i in sort_attr_set:
            f2.write(i + '\n')

    for dic_type, name_type in zip([qc_dic, ans_dic], ['source', 'target']):
        all_lst = []
        for s_n in sort_story_set:
            story_lst = []
            for a_n in sort_attr_set:
                v = dic_type.get(s_n + '_' + a_n)
                if v:
                    story_lst.append(v)
                else:
                    story_lst.append([])
            assert len(story_lst) == len(sort_attr_set), 'attr num not correct'
            if story_lst == [[]*7]:
                all_lst.append([])
            else:
                all_lst.append(story_lst)
        assert len(all_lst) == len(sort_story_set), 'story num not correct'

        print(f'all num: {len(all_lst)}')
        print('d')

        if mode in ['skill', 'qa']:
            if not os.path.exists(f'{data_src}/{mode}'):
                os.mkdir(f'{data_src}/{mode}')
            with open(f'{data_src}/{mode}/{split_type}.{name_type}', 'w', encoding='utf-8') as f:
                for i in all_lst:
                    for j in i[:7]:
                        for k in j:
                            f.write(k + '\n')
            print(f'save: {data_src}/{mode}/{split_type}.{name_type}')
        else:
            exim = 0
            if mode == 'exim':
                exim = 1
                mode = 'cl'
            
            if not os.path.exists(f'{data_src}/{mode}'):
                os.mkdir(f'{data_src}/{mode}')
            with open(f'{data_src}/{mode}/{split_type}_{name_type}.pkl', 'wb') as f:
                pickle.dump(all_lst, f)
            # with open(f'{mode}_{name_type}.pkl', 'rb') as f:
            #     data = pickle.load(f)
            print(f'save: {data_src}/{mode}/{split_type}_{name_type}.pkl')
            
            if exim:
                attr_list = ['explicit', 'implicit']
                for name in ['train', 'val', 'test']:
                    fs = open(f'{data_src}/cl/{name}_source.pkl', 'rb')
                    data_s = pickle.load(fs)
                    os.makedirs(f'{data_src}/exim', exist_ok=True)
                    with open(f'{data_src}/exim/{name}.source', 'w+', encoding='utf-8') as f1:
                        for story in data_s:
                            for ex_im, attr in zip(story[-2:], attr_list):
                                for content in ex_im:
                                    f1.write(attr + ' <SEP> ' + content + '\n')

                    ft = open(f'{data_src}/cl/{name}_target.pkl', 'rb')
                    data_t = pickle.load(ft)
                    with open(f'{data_src}/exim/{name}.target', 'w+', encoding='utf-8') as f2:
                        for story in data_t:
                            for ex_im in story[-2:]:
                                for content in ex_im:
                                    f2.write(content + '\n')
print('d')


# python process.py qa all fairy
