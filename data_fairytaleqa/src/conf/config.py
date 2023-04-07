#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number

"""
Author: Wesley (liwanshui12138@gmail.com)
Date: 2022-5-6
"""
import os


class Config(object):
    def __init__(self):
        conf_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.dirname(os.path.dirname(conf_dir))
        self.data_dir = os.path.join(self.work_dir, 'data')

