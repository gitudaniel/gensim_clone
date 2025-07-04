#!/usr/bin/env python
# encoding: utf-8

from collections import namedtuple
import unittest
import logging

import numpy as np
import pytest

from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile


class TestTranslationMatrix(unittest.TestCase):
    def setUp(self):
        self.source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        self.target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")

        self.word_pairs = [
            ("one", "uno"), ("two", "due"), ("three", "tre"),
            ("four", "quattro"), ("five", "cinque"), ("seven", "sette"), ("eight", "otto"),
            ("dog", "cane"), ("pig", "maiale"), ("fish", "cavallo"), ("birds", "uccelli"),
            ("apple", "mela"), ("orange", "arancione"), ("grape", "acino"), ("banana", "banana"),
        ]

        self.test_word_pairs = [("ten", "dieci"), ("cat", "gatto")]

        self.source_word_vec = KeyedVectors.load_word2vec_format(self.source_word_vec_file, binary=False)
        self.target_word_vec = KeyedVectors.load_word2vec_format(self.target_word_vec_file, binary=False)

    def test_translation_matrix(self):
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        self.assertEqual(model.translation_matrix.shape, (300, 300))

    def test_persistence(self):
        """Test storing/loading the entire model."""
        tmpf = get_tmpfile('transmat-en-it.pkl')

        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        model.save(tmpf)

        loaded_model = translation_matrix.TranslationMatrix.load(tmpf)
        self.assertTrue(np.allclose(model.translation_matrix, loaded_model.translation_matrix))

    def test_translate_nn(self):
        # Test the nearest neighbor retrieval method
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)

        test_source_word, test_target_word = zip(*self.test_word_pairs)
        translated_words = model.translate(
            test_source_word, topn=5, source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec,
        )

        for idx, item in enumerate(self.test_word_pairs):
            self.assertTrue(item[1] in translated_words[item[0]])

    @pytest.mark.xfail(
        True,
        reason='blinking test, can be related to <https://github.com/RaRe-Technologies/gensim/issues/2977>'
    )
    def test_translate_gc(self):
        # Test globally corrected neighbor retrieval method
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)

        test_source_word, test_target_word = zip(*self.test_word_pairs)
        translated_words = model.translate(
            test_source_word, topn=5, gc=1, sample_num=3,
            source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec
        )

        for idx, item in enumerate(self.test_word_pairs):
            self.assertTrue(item[1] in translated_words[item[0]])


def read_sentiment_docs(filename):
    sentiment_document = namedtuple('SentimentDocument', 'words tags')
    alldocs = []  # will hold all docs in original order
    with utils.open(filename, mode='rb', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = utils.to_unicode(line).split()
            words = tokens
            tags = str(line_no)
            alldocs.append(sentiment_document(words, tags))
    return alldocs


class TestBackMappingTranslationMatrix(unittest.TestCase):
    def setUp(self):
        filename = datapath("alldata-id-10.txt")
        train_docs = read_sentiment_docs(filename)
        self.train_docs = train_docs
        self.source_doc_vec = Doc2Vec(documents=train_docs[:5], vector_size=8, epochs=50, seed=1)
        self.target_doc_vec = Doc2Vec(documents=train_docs, vector_size=8, epochs=50, seed=2)

    def test_translation_matrix(self):
        model = translation_matrix.BackMappingTranslationMatrix(
            self.source_doc_vec, self.target_doc_vec, self.train_docs[:5],
        )
        transmat = model.train(self.train_docs[:5])
        self.assertEqual(transmat.shape, (8, 8))

    @unittest.skip(
        "flaky test likely to be discarded when <https://github.com/RaRe-Technologies/gensim/issues/2977> "
        "is addressed"
    )
    def test_infer_vector(self):
        """Test that translation gives similar results to traditional inference.

        This may not be completely sensible/salient with such tiny data, but
        replaces what seemed to me to be an ever-more-nonsensical test.

        See <https://github.com/RaRe-Technologies/gensim/issues/2977> for discussion
        of whether the class this supposedly tested even survives when the
        TranslationMatrix functionality is better documented.
        """
        model = translation_matrix.BackMappingTranslationMatrix(
            self.source_doc_vec, self.target_doc_vec, self.train_docs[:5],
        )
        model.train(self.train_docs[:5])
        backmapped_vec = model.infer_vector(self.target_doc_vec.dv[self.train_docs[5].tags[0]])
        self.assertEqual(backmapped_vec.shape, (8, ))

        d2v_inferred_vector = self.source_doc_vec.infer_vector(self.train_docs[5].words)

        distance = cosine(backmapped_vec, d2v_inferred_vector)
        self.assertLessEqual(distance, 0.1)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
