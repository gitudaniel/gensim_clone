#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest

import numpy as np

import scipy.linalg

from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile


class TestLsiModel(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))
        self.model = lsimodel.LsiModel(self.corpus, num_topics=2)
