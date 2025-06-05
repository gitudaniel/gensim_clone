#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Automatically detect common phrases -- aka multi-word expressions, word n-gram collocations -- from
a stream of sentences.

Inspired by:

* `Mikolov, et. al: "Distributed Representations of Words and Phrases and their Compositionality"
  <https://arxiv.org/abs/1310.4546>`_
* `"Normalized (Pointwise) Mutual Information in Collocation Extraction" by Gerlof Bouma
  <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_


Examples
--------
.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>> from gensim.models.word2vec import Text8Corpus
    >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    >>>
    >>> # Create training corpus. Must be a sequence of sentences (e.g. an iterable or a generator).
    >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
    >>> # Each sentence must be a list of string tokens:
    >>> first_sentence = next(iter(sentences))
    >>> print(first_sentence[:10])
    ['computer', 'human', 'interface', 'computer', 'response', 'survey', 'system', 'time', 'user', 'interface']
    >>>
    >>> # Train a toy phrase model on our training corpus.
    >>> phrase_model = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
    >>>
    >>> # Apply the trained phrases model to a new, unseen sentence.
    >>> new_sentence = ['trees', 'graph', 'minors']
    >>> phrase_model[new_sentence]
    ['trees_graph', 'minors']
    >>> # The toy model considered "trees graph" a single phrase => joined the two
    >>> # tokens into a single "phrase" token, using our selected `_` delimiter.
    >>>
    >>> # Apply the trained model to each sentence of a corpus, using the same [] syntax:
    >>> for sent in phrase_model[sentences]:
    ...     pass
    >>>
    >>> # Update the model with two new sentences on the fly.
    >>> phrase_model.add_vocab([["hello", "world"], ["meow"]])
    >>>
    >>> # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    >>> frozen_model = phrase_model.freeze()
    >>> # Apply the frozen model; same results as before:
    >>> frozen_model[new_sentence]
    ['trees_graph', 'minors']
    >>>
    >>> # Save / load models.
    >>> frozen_model.save("/tmp/my_phrase_model.pkl")
    >>> model_reloaded = Phrases.load("/tmp/my_phrase_model.pkl")
    >>> model_reloaded[['trees', 'graph', 'minors']]  # apply the reloaded model to a sentence
    ['trees_graph', 'minors']

"""

import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time

from gensim import utils, interfaces


logger = logging.getLogger(__name__)

NEGATIVE_INFINITY = float('-inf')

# Words from this set are "ignored" during phrase detection:
# 1) Phrases may not start nor end with these words.
# 2) Phrases may include any number of these words inside.
ENGLISH_CONNECTOR_WORDS = frozenset(
    " a an the "  # articles; we never care about these in MWEs
    " for of with without at from to in on by "  # prepositions; incomplete on purpose, to minimize FNs
    " and or "  # conjunctions; incomplete on purpose, to minimize FNs
    .split()
)


def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    r"""Bigram scoring function, based on the original `Mikolov, et. al: "Distributed Representations
    of Words and Phrases and their compositionality" <https://arxiv.org/abs/1310.4546>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Size of vocabulary.
    min_count : int
        Minimum collocation count threshold.
    corpus_word_count : int
        Not used in this particular scoring technique.

    Returns
    -------
    float
        Score for given phrase. Can be negative.

    Notes
    -----
    Formula: :math:`\frac{(bigram\_count - min\_count * len\_vocab }{ (worda\_count * wordb\_count)}`.

    """
    denom = worda_count * wordb_count
    if denom == 0:
        return NEGATIVE_INFINITY
    return (bigram_count - min_count) / float(denom) * len_vocab


def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    r"""Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"
    by Gerlof Bouma <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurences for phrase "worda_wordb".
    len_vocab : int
        Not used.
    min_count : int
        Ignore all bigrams with total collected count lower than this value.
    corpus_word_count : int
        Total number of words in the corpus.

    Returns
    -------
    float
        If bigram_count >= min_count, return the collocation score, in the range -1 to 1.
        Otherwise return -inf.

    Notes
    -----
    Formula: :math:`\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,
    where :math:`prob(word) = \frac{word\_count}{corpus\_word\_count}`

    """
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        try:
            return log(pab / (pa * pb)) / -log(pab)
        except ValueError:  # some of the counts were zero => never a phrase
            return NEGATIVE_INFINITY
    else:
        # Return -infinity to make sure that no phrases will be created
        # from bigrams less frequent than min_count.
        return NEGATIVE_INFINITY


def _is_single(obj):
    """Check whether `obj` is a single document of an entire corpus.

    Parameters
    ----------
    obj : object

    Return
    ------
    (bool, object)
        2-tuple ``(is_single_document, new_obj)`` tuple, where `new_obj`
        yields the same sequence as the original `obj`.

    Notes
    -----
    `obj` is a single document if it is an iterable of strings. It is a corpus if it is an iterable of documents.

    """
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = itertools.chain([peek], obj_iter)
    except StopIteration:
        # An empty object is interpreted as a single document (not a corpus).
        return True, obj
    if isinstance(peek, str):
        # First item is a string => obj is a single document for sure.
        return True, obj_iter
    if temp_iter is obj:
        # An iterator / generator => interpret input as a corpus.
        return False, obj_iter
    # If the first item isn't a string, assume obj is an iterable corpus.
    return False, obj


class _PhrasesTransformation(interfaces.TransformationABC):
    """
    Abstract base class for :class:`~gensim.models.phrases.Phrases` and
    :class:`~gensim.models.phrases.FrozenPhrases`.

    """
    def __init__(self, connector_words):
        self.connector_words = frozenset(connector_words)

    def score_candidate(self, word_a, word_b, in_between):
        """Score a single phrase candidate.

        Returns
        -------
        (str, float)
            2-tuple of ``(delimiter-joined phrase, phrase score)`` for a phrase,
            or ``(None, None)`` if not a phrase.
        """
        raise NotImplementedError("ABC: override this method in child classes")
    
    def analyze_sentence(self, sentence):
        """Analyze a sentence, concatenating any detected phrases into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.

        Yields
        ------
        (str, {float, None})
            Iterate through the input sentence tokens and yield 2-tuples of:
            - ``(concatenated_phrase_tokens, score)`` for token sequences that form a phrase.
            - ``(word, None)`` if the token is not a part of a phrase.

        """
        start_token, in_between = None, []
        for word in sentence:
            if word not in self.connector_words:
                # The current word is a normal token, not a connector word, which means it's a potential
                # beginning (or end) of a phrase.
                if start_token:
                    # We're inside a potential phrase, of which this word is the end.
                    phrase, score = self.score_candidate(start_token, word, in_between)
                    if score is not None:
                        # Phrase detected!
                        yield phrase, score
                        start_token, in_between = None, []
                    else:
                        # Not a phrase after all. Dissolve the candidate's constituent tokens as individual words.
                        yield start_token, None
                        for w in in_between:
                            yield w, None
                        start_token, in_between = word, []  # new potential phrase starts here
                else:
                    # Not inside a phrase yet; start a new phrase candidate here.
                    start_token, in_between = word, []
            else:  # We're a connector word.
                if start_token:
                    # We're inside a potential phrase: add the connector word and keep growing the phrase.
                    in_between.append(word)
                else:
                    # Not inside a phrase: emit the connector word and move on.
                    yield word, None
        # Emit any non-phrase tokens at the end.
        if start_token:
            yield start_token, None
            for w in in_between:
                yield w, None

    def __getitem__(self, sentences):
