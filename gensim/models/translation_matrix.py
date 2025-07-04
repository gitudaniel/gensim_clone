#!/usr/bin/env python
# encoding: utf-8

"""Produce a translation matrix to translate words from one language to another, using either
a standard nearest neighbor method or a globally corrected neighbor retrieval method [1]_.

This method can be used to augment the existing phrase tables with more candidate translations, or
filter out errors from the translation tables and known dictionaries [2]_. What's more, it also works
for any two sets of named-vectors where there are some paired-guideposts to learn the transformation.

Examples
--------

How to make translations between two sets of word-vectors
=========================================================

Initialize two word-vector models

.. sourcecode:: pycon

    >>> from gensim.models import KeyedVectors
    >>> from gensim.test.utils import datapath
    >>>
    >>> model_en = KeyedVectors.load_word2vec_format(datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"))
    >>> model_it = KeyedVectors.load_word2vec_format(datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt"))

Define word pairs (that will be used for construction of translation matrix)

.. sourcecode:: pycon

    >>> word_pairs = [
    ...     ("one", "uno"), ("two", "due"), ("three", "tre"), ("four", "quattro"), ("five", "cinque"),
    ...     ("seven", "sette"), ("eight", "otto"),
    ...     ("dog", "cane"), ("pig", "maiale"), ("fish", "cavallo"), ("birds", "uccelli"),
    ...     ("apple", "mela"), ("orange", "arancione"), ("grape", "acino"), ("banana", "banana")
    ... ]

Fit :class:`~gensim.models.translation_matrix.TranslationMatrix`

.. sourcecode:: pycon

    >>> trans_model = TranslationMatrix(model_en, model_it, word_pairs=word_pairs)

Apply model (translate words "dog" and "one")

.. sourcecode:: pycon

    >>> trans_model.translate(["dog", "one"], topn=3)
    OrderedDict([('dog', [u'cane', u'gatto', u'cavallo']), ('one', [u'uno', u'due', u'tre'])])


Save / load model

.. sourcecode:: pycon

    >>> with temporary_file("model_file") as fname:
    ...     trans_model.save(fname)  # save model to file
    ...     loaded_trans_model = TranslationMatrix.load(fname)  # load model


How to make translation between two :class:`~gensim.models.doc2vec.Doc2Vec` models
==================================================================================

Prepare data and models

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>> from gensim.test.test_translation_matrix import read_sentiment_docs
    >>> from gensim.models import Doc2Vec
    >>>
    >>> data = read_sentiment_docs(datapath("alldata-id-10.txt"))[:5]
    >>> src_model = Doc2Vec.load(datapath("small_tag_doc_5_iter50"))
    >>> dst_model = Doc2Vec.load(datapath("large_tag_doc_10_iter50"))

Train backward translation

.. sourcecode:: pycon

    >>> model_trans = BackMappingTranslationMatrix(data, src_model, dst_model)
    >>> trans_matrix = model_trans.train(data)


Apply model

.. sourcecode:: pycon

    >>> result = model_trans.infer_vector(dst_model.dv[data[3].tags])


References
----------
.. [1] Dinu, Georgiana, Angeliki Lazaridou, and Marco Baroni. "Improving zero-shot learning by mitigating the
       hubness problem", https://arxiv.org/abs/1412.6568
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       "Distributed Representations of Words and Phrases and their Compositionality", https://arxiv.org/abs/1310.4546

"""

import warnings
from collections import OrderedDict

import numpy as np

from gensim import utils


class Space:
    """An auxiliary class for storing the word space."""

    def __init__(self, matrix, index2word):
        """
        Parameters
        ----------
        matrix : iterable of numpy.ndarray
            Matrix that contains word-vectors.
        index2word : list of str
            Words which correspond to the `matrix`.

        """
        self.mat = matrix
        self.index2word = index2word

        # build a dict to map word to index
        self.word2index = {}
        for idx, word in enumerate(self.index2word):
            self.word2index[word] = idx

    @classmethod
    def build(cls, lang_vec, lexicon=None):
        """Construct a space class for the lexicon, if it's provided.

        Parameters
        ----------
        lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`
            Model from which the vectors will be extracted.
        lexicon : list of str, optional
            Words contained in the `lang_vec`, if `lexicon = None`, the lexicon is all the lang_vec's words.

        Returns
        -------
        :class:`~gensim.models.translation_matrix.Space`
            Object that stored word-vectors.

        """
        # `words` to store all the words
        # `mat` to store the word vector for each word in the 'words' list
        words = []
        mat = []
        if lexicon is not None:
            # if the lexicon is not provided, using all the KeyedVector's words as default
            for item in lexicon:
                words.append(item)
                mat.append(lang_vec.vectors[lang_vec.get_index(item)])

        else:
            for item in lang_vec.index_to_key:
                words.append(item)
                mat.append(lang_vec.vectors[lang_vec.get_index(item)])

        return Space(mat, words)

    def normalize(self):
        """Normalize the word vectors matrix."""
        self.mat = self.mat / np.sqrt(np.sum(np.square(self.mat), axis=1, keepdims=True))


class TranslationMatrix(utils.SaveLoad):
    """Objects of this class realize the translation matrix which maps the source language to the target language.
    The main methods are:

    We map it to the other language space by computing z = Wx, then return the
    word whose representation is close to z.

    For details on use, see the tutorial notebook [3]_

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.models import KeyedVectors
        >>> from gensim.test.utils import datapath
        >>> en = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        >>> it = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        >>> model_en = KeyedVectors.load_word2vec_format(en)
        >>> model_it = KeyedVectors.load_word2vec_format(it)
        >>>
        >>> word_pairs = [
        ...     ("one", "uno"), ("two", "due"), ("three", "tre"), ("four", "quattro"), ("five", "cinque"),
        ...     ("seven", "sette"), ("eight", "otto"),
        ...     ("dog", "cane"), ("pig", "maiale"), ("fish", "cavallo"), ("birds", "uccelli"),
        ...     ("apple", "mela"), ("orange", "arancione"), ("grape", "acino"), ("banana", "banana")
        ... ]
        >>>
        >>> trans_model = TranslationMatrix(model_en, model_it)
        >>> trans_model.train(word_pairs)
        >>> trans_model.translate(["dog", "one"], topn=3)
        OrderedDict([('dog', [u'cane', u'gatto', u'cavallo']), ('one', [u'uno', u'due', u'tre'])])


    References
    ----------
    .. [3] https://github.com/RaRe-Technologies/gensim/blob/3.2.0/docs/notebooks/translation_matrix.ipynb

    """
    def __init__(self, source_lang_vec, target_lang_vec, word_pairs=None, random_state=None):
        """
        Parameters
        ----------
        source_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`
            Word vectors for source language.
        target_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`
            Word vectors for target language.
        word_pairs : list of (str, str), optional
            Pairs of words that will be used for training.
        random_state : {None, int, array_like}, optional
            Seed for random state.

        """

        self.source_word = None
        self.target_word = None
        self.source_lang_vec = source_lang_vec
        self.target_lang_vec = target_lang_vec

        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None
        self.source_space = None
        self.target_space = None

        if word_pairs is not None:
            if len(word_pairs[0]) != 2:
                raise ValueError("Each training data item must contain two different language words.")
            self.train(word_pairs)

    def train(self, word_pairs):
        """Build the translation matrix to map from source space to target space.

        Parameters
        ----------
        word_pairs : list of (str, str), optional
            Pairs of words that will be used for training.

        """
        self.source_word, self.target_word = zip(*word_pairs)

        self.source_space = Space.build(self.source_lang_vec, set(self.source_word))
        self.target_space = Space.build(self.target_lang_vec, set(self.target_word))

        self.source_space.normalize()
        self.target_space.normalize()

        m1 = self.source_space.mat[[self.source_space.word2index[item] for item in self.source_word], :]
        m2 = self.target_space.mat[[self.target_space.word2index[item] for item in self.target_word], :]

        self.translation_matrix = np.linalg.lstsq(m1, m2, -1)[0]

    def save(self, *args, **kwargs):
        """Save the model to a file. Ignores (doesn't store) the `source_space` and `target_space` attributes."""
        kwargs['ignore'] = kwargs.get('ignore', ['source_space', 'target_space'])
        super(TranslationMatrix, self).save(*args, **kwargs)

    def apply_transmat(self, words_space):
        """Map the source word vector to the target word vector using translation matrix.

        Parameters
        ----------
        words_space : :class:`~gensim.models.translation_matrix.Space`
            `Space` object constructed for the words to be translated.

        Returns
        -------
        :class:`~gensim.models.translation_matrix.Space`
            `Space` object constructed for the mapped words.

        """
        return Space(np.dot(words_space.mat, self.translation_matrix), words_space.index2word)

    def translate(self, source_words, topn=5, gc=0, sample_num=None, source_lang_vec=None, target_lang_vec=None):
        """Translate the word from the source language to the target language.

        Parameters
        ----------
        source_words : {str, list of str}
            Single word or a list of words to be translated.
        topn : int, optional
            Number of words that will be returned as translation for each `source_words`
        gc : int, optional
            Define the translation algorithm, if `gc == 0` -use standard NN retrieval,
            otherwise, use globally corrected neighbor retrieval method (as described in [1]_).
        sample_num : int, optional
            Number of words to sample from the source lexicon, if `gc == 1`, then `sample_num` **must** be provided.
        source_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
            New source language vectors for translation, by default, used the model's source language vector.
        target_lang_vec : :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
            New target language vectors for translation, by default, used the model's target language vector.

        Returns
        -------
        :class:`collections.OrderedDict`
            Ordered dict where each item is `word`: [`translated_word_1`, `translated_word_2`, ...]

        """
        if isinstance(source_words, str):
            # pass only one word to translate
            source_words = [source_words]

        # If the language word vector is not provided by user, use the model's
        # language word vector as default
        if source_lang_vec is None:
            warnings.warn(
                "The parameter source_lang_vec isn't specified, "
                "use the model's source language word vector as default."
            )
            source_lang_vec = self.source_lang_vec

        if target_lang_vec is None:
            warnings.warn(
                "The parameter target_lang_vec isn't specified, "
                "use the model's target language word vector as default."
            )
            target_lang_vec = self.target_lang_vec

        # If additional is provided, bootstrapping vocabulary from the source language word vector model.
        if gc:
            if sample_num is None:
                raise RuntimeError(
                    "When using the globally corrected neighbor retrieval method, "
                    "the `sample_num` parameter (i.e. the number of words sampled from source space) must be provided."
                )
            lexicon = set(source_lang_vec.index_to_key)
            addition = min(sample_num, len(lexicon) - len(source_words))
            lexicon = self.random_state.choice(list(lexicon.difference(source_words)), addition)
            source_space = Space.build(source_lang_vec, set(source_words).union(set(lexicon)))
        else:
            source_space = Space.build(source_lang_vec, source_words)
        target_space = Space.build(target_lang_vec, )

        # Normalize the source vector and target vector
        source_space.normalize()
        target_space.normalize()

        # Map the source language to the target language
        mapped_source_space = self.apply_transmat(source_space)

        # Use the cosine similarity metric
        sim_matrix = -np.dot(target_space.mat, mapped_source_space.mat.T)

        # If `gc=1`, using corrected retrieval method
        if gc:
            srtd_idx = np.argsort(np.argsort(sim_matrix, axis=1), axis=1)
            sim_matrix_idx = np.argsort(srtd_idx + sim_matrix, axis=0)
        else:
            sim_matrix_idx = np.argsort(sim_matrix, axis=0)

        # Translate the words and for each word return the `topn` similar words
        translated_word = OrderedDict()
        for idx, word in enumerate(source_words):
            translated_target_word = []
            # Search the most `topn` similar words
            for j in range(topn):
                map_space_id = sim_matrix_idx[j, source_space.word2index[word]]
                translated_target_word.append(target_space.index2word[map_space_id])
            translated_word[word] = translated_target_word
        return translated_word


class BackMappingTranslationMatrix(utils.SaveLoad):
    """Realize the BackMapping translation matrix which maps the source model's document vector
    to the target model's document vector (old model).

    BackMapping translation matrix is used to learn a mapping for two document vector spaces which we
    specify as source document vector and target document vector. The target document vectors are trained
    on a superset corpus of source document vectors; we can incrementally increase the vector in
    the old model through the BackMapping translation matrix.

    For details on use, see the tutorial notebook [3]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>> from gensim.test.test_translation_matrix import read_sentiment_docs
        >>> from gensim.models import Doc2Vec, BackMappingTranslationMatrix
        >>>
        >>> data = read_sentiment_docs(datapath("alldata-id-10.txt"))[:5]
        >>> src_model = Doc2Vec.load(datapath("small_tag_doc_5_iter50"))
        >>> dst_model = Doc2Vec.load(datapath("large_tag_doc_10_iter50"))
        >>>
        >>> model_trans = BackMappingTranslationMatrix(src_model, dst_model)
        >>> trans_matrix = model_trans.train(data)
        >>>
        >>> result = model_trans.infer_vector(dst_model.dv[data[3].tags])

    """
    def __init__(self, source_lang_vec, target_lang_vec, tagged_docs=None, random_state=None):
        """

        Parameters
        ----------
        source_lang_vec : :class:`~gensim.models.doc2vec.Doc2Vec`
            Source Doc2Vec model.
        target_lang_vec : :class:`~gensim.models.doc2vec.Doc2Vec`
            Target Doc2Vec model.
        tagged_docs : list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            Documents that will be used for training, both the source language document vector and
            target language document vector trained on those tagged documents.
        random_state : {None, int, array_like}, optional
            Seed for random state.

        """
        self.tagged_docs = tagged_docs
        self.source_lang_vec = source_lang_vec
        self.target_lang_vec = target_lang_vec

        self.random_state = utils.get_random_state(random_state)
        self.translation_matrix = None

        if tagged_docs is not None:
            self.train(tagged_docs)

    def train(self, tagged_docs):
        """Build the translation matrix to map from the source model's vectors to target model's vectors

        Parameters
        ----------
        tagged_docs : list of :class:`~gensim.models.doc2vec.TaggedDocument`
            Documents that will be used for training, both the source language document vector and
            target language document vector trained on those tagged documents.

        Returns
        -------
        numpy.ndarray
            Translation matrix that maps from the source model's vectors to target model's vectors.

        """
        m1 = [self.source_lang_vec.dv[item.tags].flatten() for item in tagged_docs]
        m2 = [self.target_lang_vec.dv[item.tags].flatten() for item in tagged_docs]

        self.translation_matrix = np.linalg.lstsq(m2, m1, -1)[0]
        return self.translation_matrix

    def infer_vector(self, target_doc_vec):
        """Translate the target model's document vector to the source model's document vector

        Parameters
        ----------
        target_doc_vec : numpy.ndarray
            Document vector from the target document, whose documents are not in the source model.

        Returns
        -------
        numpy.ndarray
            Vector `target_doc_vec` in the source model.

        """
        return np.dot(target_doc_vec, self.translation_matrix)
