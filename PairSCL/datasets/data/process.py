#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number

"""
Author: Wesley (liwanshui12138@gmail.com)
Date: 2022/6/9
"""
import pickle

key = "skill"

if key == "exim":
    attr_list = ['explicit', 'implicit']

    for name in ['train', 'val', 'test']:
        fs = open(f'{name}_source.pkl', 'rb')
        data_s = pickle.load(fs)
        with open(f'exim/{name}.source', 'w+', encoding='utf-8') as f1:
            for story in data_s:
                for ex_im, attr in zip(story[-2:], attr_list):
                    for content in ex_im:
                        f1.write(attr + ' <SEP> ' + content + '\n')

        # ft = open(f'{name}_target.pkl', 'rb')
        # data_t = pickle.load(ft)
        # with open(f'exim/{name}.target', 'w+', encoding='utf-8') as f2:
            # for story in data_t:
                # for ex_im in story[-2:]:
                    # for content in ex_im:
                        # f2.write(content + '\n')

elif key == "skill":
    attr_list = ['action', 'causal relationship', 'character', 'feeling', 'outcome resolution', 'prediction', 'setting']

    for name in ['train', 'val', 'test']:
        fs = open(f'{name}_source.pkl', 'rb')
        data_s = pickle.load(fs)
        with open(f'skill/{name}.source', 'w+', encoding='utf-8') as f1:
            for story in data_s:
                for ex_im, attr in zip(story[:-2], attr_list):
                    for content in ex_im:
                        f1.write(attr + ' <SEP> ' + content + '\n')

        # ft = open(f'{name}_target.pkl', 'rb')
        # data_t = pickle.load(ft)
        # with open(f'skill/{name}.target', 'w+', encoding='utf-8') as f2:
            # for story in data_t:
                # for ex_im in story[:-2]:
                    # for content in ex_im:
                        # f2.write(content + '\n')

print('d')
