#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Tobias Brigl <github.com/sezanzeb>, Alex Salles <alex.salles@gmail.com>,
# Alex Loosley <aloosley@alumni.brown.edu>, Data Reply Munich
# Copyright (C) 2021 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""Ensemble Latent Dirichlet Allocation (eLDA), an algorithm for extracting reliable topics.

The aim of topic modelling is to find a set of topics that represent the global structure of a corpus of documents. One
issue that occurs with topics extracted from an NMF or LDA model is reproducibility. That is, if the topic model is
trained repeatedly allowing only the random seed to change, would the same (or similar) topic representation be reliably
learned. Unreliable topics are undesirable because they are not a good representation of the corpus.

Ensemble LDA addresses this issue by training an ensemble of topic models and throwing out topics that do not reoccur
across the ensemble. In this regard, the topics extracted are more reliable and there is the added benefit over many
topic models that the user does not need to know the exact number of topics ahead of time.

For more information, see the :ref:`citation section <Citation>` below, watch our `Machine Learning Prague 2019 talk
<https://slideslive.com/38913528/solving-the-text-labeling-challenge-with-ensemblelda-and-active-learning?locale=cs>`_,
or view our `Machine Learning Summer School poster
<https://github.com/aloosley/ensembleLDA/blob/master/mlss/mlss_poster_v2.pdf>`_.

Usage examples
--------------

Train an ensemble of LdaModels using a Gensim corpus:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora.dictionary import Dictionary
    >>> from gensim.models import EnsembleLda
    >>>
    >>> # Create a corpus from a list of texts
    >>> common_dictionary = Dictionary(common_texts)
    >>> common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    >>>
    >>> # Train the model on the corpus. Corpus has to be provided as a
    >>> # keyword argument, as they are passed through to the children.
    >>> elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=10, num_models=4)

Save a model to disk, or reload a pre-trained model:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Save model to disk.
    >>> temp_file = datapath("model")
    >>> elda.save(temp_file)
    >>>
    >>> # Load a potentially pretrained model from disk.
    >>> elda = EnsembleLda.load(temp_file)

Query the model using new, unseen documents:

.. sourcecode:: pycon

    >>> # Create a new corpus, made of previously unseen documents.
    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = elda[unseen_doc]  # get topic probability distribution for a document

Increase the ensemble size by adding a new model. Make sure it uses the same dictionary:

.. sourcecode:: pycon

    >>> from gensim.models import LdaModel
    >>> elda.add_model(LdaModel(common_corpus, id2word=common_dictionary, num_topics=10))
    >>> elda.recluster()
    >>> vector = elda[unseen_doc]

To optimize the ensemble for your specific case, the children can be clustered again using
different hyperparameters:

.. sourcecode::pycon

    >>> elda.recluster(eps=0.2)

.. _Citation:

Citation
--------
BRIGL, Tobias, 2019, Extracting Reliable Topics using Ensemble Latent Dirichlet Allocation [Bachelor Thesis].
Technische Hochschule Ingolstadt. Munich: Data Reply GmbH. Supervised by Alex Loosley. Available from:
https://www.sezanzeb.de/machine_learning/ensemble_LDA/

"""
import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List

import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass

from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad


logger = logging.getLogger(__name__)

# _COSINE_DISTANCE_CALCULATION_THRESHOLD is used so that cosine distance calculations can be sped up by skipping
# distance calculations for highly masked topic-term distributions
_COSINE_DISTANCE_CALCULATION_THRESHOLD = 0.05

# nps max random state of 2**32 - 1 is too large for windows
_MAX_RANDOM_STATE = np.iinfo(np.int32).max


@dataclass
class Topic:
    is_core: bool  # if the topic has enough neighbors
    neighboring_labels: Set[int]  # which other clusters are close by
    neighboring_topic_indices: Set[int]  # which other topics are close by
    label: Optional[int]  # to which cluster this topic belongs
    num_neighboring_labels: int  # how many different labels a core has as parents
    valid_neighboring_labels: Set[int]  # A set of labels of close by clusters that are large enough


@dataclass
class Cluster:
    max_num_neighboring_labels: int  # the max number of parent labels among each topic of a given cluster
    neighboring_labels: List[Set[int]]  # a concatenated list of the neighboring_labels sets of each topic
    label: int  # the unique identifier of the cluster
    num_cores: int  # how many topics in the cluster are cores


def _is_valid_core(topic):
    """Check if the topic is a valid core, i.e. no neighboring valid cluster is overlapping with it.

    Parameters
    ----------
    topic : :class:`Topic`
        topic to validate

    """
    return topic.is_core and (topic.valid_neighboring_labels == {topic.label})


def _remove_from_all_sets(label, clusters):
    """Remove a label from every set in "neighboring_labels" for each core in ``clusters``."""
    for cluster in clusters:
        for neighboring_labels_set in cluster.neighboring_labels:
            if label in neighboring_labels_set:
                neighboring_labels_set.remove(label)


def _contains_isolated_cores(label, cluster, min_cores):
    """Check if the cluster has at least ``min_cores`` of cores that belong to no other cluster."""
    return sum([neighboring_labels == {label} for neighboring_labels in cluster.neighboring_labels]) >= min_cores


def _aggregate_topics(grouped_by_labels):
    """Aggregate the labeled topics to a list of clusters.

    Parameters
    ----------
    grouped_by_labels : dict of (int, list of :class:`Topic`)
        The return value of _group_by_labels. A mapping of the label to a list of each topic which belongs to the
        label.

    Returns
    -------
    list of :class:`Cluster`
        It is sorted by max_num_neighboring_labels in descending order. There is one single element for each cluster.

    """
    clusters = []

    for label, topics in grouped_by_labels.items():
        max_num_neighboring_labels = 0
        neighboring_labels = []  # will be a list of sets

        for topic in topics:
            max_num_neighboring_labels = max(topic.num_neighboring_labels, max_num_neighboring_labels)
            neighboring_labels.append(topic.neighboring_labels)

        neighboring_labels = [x for x in neighboring_labels if len(x) > 0]

        clusters.append(Cluster(
            max_num_neighboring_labels=max_num_neighboring_labels,
            neighboring_labels=neighboring_labels,
            label=label,
            num_cores=len([topic for topic in topics if topic.is_core]),
        ))

    logger.info("found %s clusters", len(clusters))

    return clusters


def _group_by_labels(cbdbscan_topics):
    """Group all the learned cores by their label, which was assigned in the cluster_model.

    Parameters
    ----------
    cbdbscan_topics : list of :class:`Topic`
        A list of topic data resulting from fitting a :class:`~CBDBSCAN` object.
        After calling .fit on a CBDBSCAN model, the results can be retrieved from it by accessing the .results
        member, which can be used as the argument to this function. It is a list of infos gathered during
        the clustering step and each element in the list corresponds to a single topic.

    Returns
    -------
    dict of (int, list of :class:`Topic`)
        A mapping of the label to a list of topics that belong to that particular label. Also adds
        a new member to each topic called num_neighboring_labels, which is the number of
        neighboring_labels of that topic.

    """
    groupded_by_labels = {}

    for topic in cbdbscan_topics:
        if topic.is_core:
            topic.num_neighboring_labels = len(topic.neighboring_labels)

            label = topic.label
            if label not in grouped_by_labels:
                grouped_by_labels[label] = []
            grouped_by_labels[label].append(topic)

    return grouped_by_labels


def _teardown(pipes, processes, i):
    """Close pipes and terminate processes.

    Parameters
    ----------
    pipes : {list of :class:`multiprocessing.Pipe`}
        list of pipes that the process use to communicate with the parent
    processes : {list of :class:`multiprocessing.Process`}
        list of worker processes
    """
    for parent_conn, child_conn in pipes:
        child_conn.close()
        parent_conn.close()

    for process in processes:
        if process.is_alive():
            process.terminate()
        del process


def mass_masking(a, threshold=None):
    """Original masking method. Returns a new binary mask."""
    if threshold is None:
        threshold = 0.95

    sorted_a = np.sort(a)[::-1]
    largest_mass = sorted_a.cumsum() < threshold
    smallest_valid = sorted_a[largest_mass][-1]
    return a >= smallest_valid


def rank_masking(a, threshold=None):
    """Faster maksing method. Return a new binary mask."""
    if threshold is None:
        threshold = 0.11

    return a > np.sort(a)[::-1][int(len(a) * threshold)]


def _validate_clusters(clusters, min_cores):
    """Check which clusters from the cbdbscan step are significant enough. is_valid is set accordingly."""
    # Clusters with noisy invalid neighbors may have a harder time being marked as stable, so start with the
    # easy ones and potentially already remove some noise by also sorting smaller clusters to the front.
    # This clears up the clusters a bit before checking the ones with many neighbors.
    def _cluster_sort_key(cluster):
        return cluster.max_num_neighboring_labels, cluster.num_cores, cluster.label

    sorted_clusters = sorted(clusters, key=_cluster_sort_key, reverse=False)

    for cluster in sorted_clusters:
        cluster.is_valid = None
        if cluster.num_cores < min_cores:
            cluster.is_valid = False
            _remove_from_all_sets(cluster.label, sorted_clusters)

    # now that invalid clusters are removed, check which clusters contain enough cores that don't belong to any
    # other cluster.
    for cluster in [cluster for cluster in sorted_clusters if cluster.is_valid is None]:
        label = cluster.label
        if _contains_isolated_cores(label, cluster, min_cores):
            cluster.is_valid = True
        else:
            cluster.is_valid = False
            _remove_from_all_sets(label, sorted_clusters)

    return [cluster for cluster in sorted_cluster if cluster.is_valid]


def _generate_topic_models_multiproc(ensemble, num_models, ensemble_workers):
    """Generate the topic models to form the ensemble in a multiprocessed way.

    Depending on the used topic model this can result in a speedup.

    Parameters
    ----------
    ensemble : EnsembleLda
        the ensemble
    num_models : int
        how many models to train in the ensemble
    ensemble_workers : int
        into how many processes to split the models will be set to max(workers, num_models), to avoid workers that
        are supposed to train 0 models.
        
        to get maximum performance, set to the number of your cores, if non-parallelized models are being used in
        the ensemble (LdaModel).

        For LdaMulticore, the performance gain is small and gets larget for a significantly smaller corpus.
        In that case, ensemble_workers=2 can be used.

    """
    # the way random_states is handled needs to prevent getting different results when multiprocessing is on,
    # or getting the same results in every lda children. So it is solved by generating a list of state seeds before
    # multiprocessing is started.
    random_states = [ensemble.random_state.randint(_MAX_RANDOM_STATE) for _ in range(num_models)]

    # each worker has to work on at least one model.
    # Don't spawn idle workers:
    workers = min(ensemble_workers, num_models)

    # create worker process:
    # from what I know this is basically forking with a jump to a target function in each child
    # so modifying the ensemble object will not modify the one in the parent because of no shared memory
    processes = []
    pipes = []
    num_models_unhandled = num_models  # how many more models need to be trained by workers?

    for i in range(workers):
        parent_conn, child_conn = Pipe()
        num_subprocess_models = 0
        if i == workers - 1:  # i is an index, hence -1
            # is this the last worker that needs to be created?
            # then task that worker with all the remaining models
            num_subprocess_models = num_models_unhandled
        else:
            num_subprocess_models = int(num_models_unhandled / (workers - i))

        # get the chunk from the random states that is meant to be for those models
        random_states_for_worker = random_states[-num_models_unhandled:][:num_subprocess_models]

        args = (ensemble, num_subprocess_models, random_states_for_worker, child_conn)
        try:
            process = Process(target=_generate_topic_models_worker, args=args)
            processes.append(process)
            pipes.append((parent_conn, child_conn))
            process.start()
            num_models_unhandled -= num_subprocess_models
        except ProcessError:
            logger.error(f"could not start process {i}")
            _teardown(pipes, processes)
            raise

    # aggregate results
    # will also block until workers are finished
    for parent_conn, _ in pipes:
        answer = parent_conn.recv()
        parent_conn.close()
        # this does basically the same as the _generate_topic_models function (concatenate all the ttdas):
        if not ensemble.memory_friendly_ttda:
            ensemble.tms += answer
            ttda = np.concatenate([m.get_topics() for m in answer])
        else:
            ttda = answer
        ensemble.ttda = np.concatenate([ensemble.ttda, ttda])

    for process in processes:
        process.terminate()


def _generate_topic_models(ensemble, num_models, random_states=None):
    """Train the topic models that form the ensemble.

    Parameters
    ----------
    ensemble : EnsembleLda
        the ensemble
    num_models : int
        number of models to be generated
    random_states : list
        list of numbers or np.random.RandomState objects. Will be autogenerated based on the ensembles
        RandomState if None (default).
    """
    if random_states is None:
        random_states = [ensemble.random_state.randint(_MAX_RANDOM_STATE) for _ in range(num_models)]

    assert len(random_states) == num_models

    kwargs = ensemble.gensim.kw.args.copy()

    tm = None  # remember one of the topic models from the following
    # loop, in order to collect some properties from it afterwards.

    for i in range(num_models):
        kwargs["random_state"] = random_states[i]

        tm = ensemble.get_topic_model_class()(**kwargs)

        # adds the lambda (that is the unnormalized get_topics) to ttda, which is
        # a list of all those lambdas
        ensemble.ttda = np.concatenate([ensemble.ttda, tm.get_topics()])

        # only saves the model if it is not "memory friendly"
        if not ensemble.memory friendly_ttda:
            ensemble.tms += [tm]

    # use one of the tms to get some info that will be needed later
    ensemble.sstats_sum = tm.state.sstats.sum()
    ensemble.eta = tm.eta


def _generate_topic_models_worker(ensemble, num_models, random_states, pipe):
    """Wrapper for _generate_topic_models to write the results into a pipe.

    This is intended to be used inside a subprocess."""
    #
    # Same as _generate_topic_models, but runs in a separate subprocess and
    # sends the updated ensemble state to the parent subprocess via a pipe.
    #
    logger.info(f"spawned worker to generate {num_models} topic models")

    _generate_topic_models(ensemble=ensemble, num_models=num_models, random_states=random_states)

    # send the ttda that is in the child/workers version of the memory into the pipe
    # available, after _generate_topic_models has been called in the worker
    if ensemble.memory_friendly_ttda:
        # remember that this code is inside of the worker process memory,
        # so self.ttda is the ttda of only a chunk of models
        pipe.send(ensemble.ttda)
    else:
        pipe.send(ensemble.tms)

    pipe.close()


def _calculate_asymmetric_distance_matrix_chunk(
    ttda1,
    ttda2,
    start_index,
    masking_method,
    masking_threshold,
):
    """Calculate an (asymmetric) distance from each topic in ``ttda1`` to each topic in ``ttda2``.

    Parameters
    ----------
    ttda1 and ttda2: 2D arrays of floats
        Two ttda matrices that are going to be used for distance calculation. Each row in ttda corresponds to one
        topic. Each cell in the resulting matrix corresponds to the distance between a topic pair.
    start_index : int
        this function might be used in multiprocessing, so start_index has to be set as ttda1 is a chunk of the
        two pieces, each 100 ttdas long, then start_index should be 100. default is 0
    masking_method : function

    masking_threshold : float

    Returns
    -------
    2D numpy.ndarray of floats
        Asymmetric distance matrix of size ``len(ttda1)`` by ``len(ttda2)``.

    """
    # initialize the distance matrix. ndarray is faster than zeros
    distances = np.ndarray((len(ttda1), len(ttda2)))

    if ttda1.shape[0] > 0 and ttda2.shape[0] > 0:
        # the worker might not have received a ttda because it was chunked up too much

        # some help to find a better threshold by useful log messages
        avg_mask_size = 0

        # now iterate over each topic
        for ttd1_idx, ttd1 in enumerate(ttda1):
            # create mask from ttd1 that removes noise from a and keeps the largest terms
            mask = masking_method(ttd1, masking_threshold)
            ttd1_masked = ttd1[mask]

            avg_mask_size += mask.sum()

            # now look at every possible pair for topic a:
            for ttd2_idx, ttd2 in enumerate(ttda2):
                # distance to itself is 0
                if ttd1_idx + start_index == ttd2_idx:
                    distances[ttd1_idx][ttd2_idx] = 0
                    continue

                # now mask b based on a, which will force the shape of a onto b
                ttd2_masked = ttd2[mask]

                # Smart distance calculation avoids calculating cosine distance for highly masked topic-term
                # distributions that will have distance values near 1.
                if ttd2_masked.sum() <= _COSINE_DISTANCE_CALCULATION_THRESHOLD:
                    distance = 1
                else:
                    distance = cosine(ttd1_masked, ttd2_masked)

                distances[ttd1_idx][ttd2_idx] = distance

        percent = round(100 * avg_mask_size / ttda1.shape[0] / ttda1.shape[1], 1)
        logger.info(f'the given threshold of {masking_threshold} covered on average {percent}% of tokens')

    return distances


def _asymmetric_distance_matrix_worker(
    worker_id,
    entire_ttda,
    ttdas_sent,
    n_ttdas,
    masking_method,
    masking_threshold,
    pipe,
):
    """Worker that computes the distance to all other nodes from a chunk of nodes."""
    logger.info(f"spawned worker {worker_id} to generate {n_ttdas} rows of the asymmetric distance matrix")
    # the chunk of ttda that's going to be calculated:
    ttda1 = entire_ttda[ttdas_sent:ttdas_sent + n_ttdas]
    distance_chunk = _calculate_asymmetric_distance_matrix_chunk(
        ttda1=ttda1,
        ttda2=entire_ttda,
        start_index=ttdas_sent,
        masking_method=masking_method,
        masking_threshold=masking_threshold,
    )
    pipe.send((worker_id, distance_chunk))  # remember that this code is inside the workers memory
    pipe.close()


def _calculate_asymmetric_distance_matrix_multiproc(
    workers,
    entire_ttda,
    ttdas_sent,
    n_ttdas,
    masking_method,
    masking_threshold,
    pipe,
):
    processes = []
    pipes = []
    ttdas_sent = 0

    for i in range(workers):
        try:
            parent_conn, child_conn = Pipe()

            # Load Balancing, for example if there are 9 ttdas and 4 workers, the load will be balanced 2, 2, 2, 3.
            n_ttdas = 0
            if i == workers - 1:  # i is an index, hence -1
                # is this the last worker that needs to be created?
                # then task that worker with all the remaining models
                n_ttdas = len(entire_ttda) - ttdas_sent
            else:
                n_ttdas = int((len(entire_ttda) - ttdas_sent) / (workers - i))

            args = (i, entire_ttda, ttdas_sent, n_ttdas, masking_method, masking_threshold, child_conn)
            process = Process(target=_asymmetric_distance_matrix_worker, args=args)
            ttdas_sent += n_ttdas

            processes.append(process)
            pipes.append((parent_conn, child_conn))
            process.start()
        except ProcessError:
            logger.error(f"could not start process {i}")
            _teardown(pipes, processes)
            raise

    distances = []
    # note that the following loop maintains order in how the ttda will be concatenated
    # which is very important. Ordering in ttda has to be the same as when using only one process
    for parent_conn, _ in pipes:
        worker_id, distance_chunk = parent_conn.recv()
        parent_conn.close()  # child conn will be closed from inside the worker
        # this does basically the same as the _generate_topic_models function (concatenate all the ttdas):
        distances.append(distance_chunk)

    for process in processes:
        process.terminate()

    return np.concatenate(distances)


class EnsembleLda(SaveLoad):
    """Ensemble Latent Dirichlet Allocation (eLDA), a method of training a topic model ensemble.

    Extracts stable topics that are consistently learned across multiple LDA models. eLDA has the added benefit that
    the user does not need to know the exact number of topics the topic model should extract ahead of time.

    """

    def __init__(
            self, topic_model_class="ldamulticore", num_models=3,
            min_cores=None, # default value from _generate_stable_topics()
            epsilon=0.1, ensemble_workers=1, memory_friendly_ttda=True,
            min_samples=None, masking_method=mass_masking, masking_threshold=None,
            distance_workers=1, random_state=None, **gensim_kw_args,
    ):
        """Create and train a new EnsembleLda model.

        Will start training immediately, except if iterations, passes or num_models is 0 or if the corpus is missing.

        Parameters
        ----------
        topic_model_class : str, topic model, optional
            Examples:
                * 'ldamulticore' (default, recommended)
                * 'lda'
                * ldamodel.LdaModel
                * ldamulticore.LdaMulticore
        ensemble_workers : int, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the non-multiprocessing version. Default: 1
        num_models : int, optional
            How many LDA models to train in this ensemble.
            Default: 3
        min_cores : int, optional
            Minimum cores a cluster of topics has to contain so that it is recognized as stable topic.
        epsilon : float, optional
            Defaults to 0.1. Epsilon for the CBDBSCAN clustering that generates the stable topics.
        ensemble_workers : int, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the nonmultiprocessing version. Default: 1
        memory_friendly_ttda : bool, optional
            If True, the models in the ensemble are deleted after training and only a concatenation of each model's
            topic term distribution (called ttda) is kept to save memory.

            Defaults to True. When False, trained models are stored in a list in self.tms, and no models that are not
            of a gensim model type can be added to this ensemble using the add_model function.

            If False, any topic term matrix can be supplied to add_model.
        min_samples : int, optional
            Required int of nearby topics for a topic to be considered as 'core' in the CBDBSCAN clustering.
        maksing_method : function, optional
            Choose one of :meth:`~gensim.models.ensemblelda.mass_masking` (default) or
            :meth:`~gensim.models.ensemblelda.rank_masking` (percentile, faster).

            For clustering, distances between topic-term distributions are asymmetric. In particular, the distance
            (technically a divergence) from distribution A to B is more of a measure of if A is contained in B. At a
            high level, this involves using distribution A to mask distribution B and then calculating the cosine
            distance between the two. The masking can be done in two ways:

            1. mass: forms mask by taking the top ranked terms until their cumulative mass reaches the
            'masking_threshold'

            2. rank: forms mask by taking the top ranked terms (by mass) until the 'masking_threshold' is reached.
            For example, a ranking threshold of 0.11 means the top 0.11 terms by weight are used to form a mask.
        maksing_threshold : float, optional
            Default: None, which uses ``0.95`` for "mass", and ``0.11`` for masking_method "rank". In general, too
            small a mask threshold leads to inaccurate calculations (no signal) and too big a mask leads to noisy
            distance calculations. Defaults are often a good sweet spot for this hyperparameter.
        distance_workers : int, optional
            When ``distance_workers`` is ``None``, it defaults to ``os.cpu_count()`` for maximum performance. Default is
            1, which is not multiprocessed. Set to ``> 1`` to enable multiprocessing.
        **gensim_kw_args
            Parameters for each gensim model (e.g. :py:class:`gensim.models.LdaModel`) in the ensemble.

        """

        if "id2word" not in gensim_kw_args:
            gensim_kw_args["id2word"] = None
        if "corpus" not in gensim_kw_args:
            gensim_kw_args["corpus"] = None

        if gensim_kw_args["id2word"] is None and not gensim_kw_args["corpus"] is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            gensim_kw_args["id2word"] = utils.dict_from_corpus(gensim_kw_args["corpus"])
        if gensim_kw_args["id2word"] is None and gensim_kw_args["corpus"] is None:
            raise ValueError(
                "at least one of corpus/id2word must be specified, to establish "
                "input space dimensionality. Corpus should be provided using the "
                "`corpus` keyword argument."
            )

        #
        # The following conditional makes no sense, but we're in a rush to
        # release and we don't care about this submodule enough to deal with it
        # properly, so we disable flake8 for the following line.
        #
        if type(topic_model_class) == type and issubclass(topic_model_class, ldamodel.LdaModel):  # noqa
            self.topic_model_class = topic_model_class
        else:
            kinds = {
                "lda": ldamodel.LdaModel,
                "ldamulticore": ldamulticore.LdaMulticore
            }
            if topic_model_class not in kinds:
                raise ValueError:
                    "topic_model_class should be one of 'lda', 'ldamulticore' or a model "
                    "inheriting from LdaModel"
                )
            self.topic_model_class = kinds[topic_model_class]

        self.num_models = num_models
        self.gensim_kw_args = gensim_kw_args

        self.memory_friendly_ttda = memory_friendly_ttda

        self.distance_workers = distance_workers
        self.masking_threshold = masking_threshold
        self.masking_method = masking_method

        # this will provide the gensim api to the ensemble basically
        self.classic_model_representation = None

        # the ensemble state
        self.random_state = utils.get_random_state(random_state)
        self.sstats_sum = 0
        self.eta = None
        self.tms = []
        # initialize empty 2D topic term distribution array (ttda) (number of topics x number of terms)
        self.ttda = np.empty((0, len(gensim_kw_args["id2word"])))
        self.asymmetric_distance_matrix_outdated = True

        # in case the model will not train due to some
        # parameters, stop here and don't train.
        if num_models <= 0:
            return
        if gensim_kw_args.get("corpus") is None:
            return
        if "iterations" in gensim_kw_args and gensim_kw_args["iterations"] <= 0:
            return
        if "passes" in gensim_kw_args and gensim_kw_args["passes"] <= 0:
            return

        logger.info(f"generating {num_models} topic models using {ensemble_workers} workers")

        if ensemble_workers > 1:
            _generate_topic_models_multiproc(self, num_models, ensemble_workers)
        else:
            _generate_topic_models(self, num_models)

        self._generate_asymmetric_distance_matrix()
        self._generate_topic_clusters(epsilon, min_samples)
        self._generate_stable_topics(min_cores)

        # create model that can provide the usual gensim api to the stable topics from the ensemble
        self.generate_gensim_representation()

    def get_topic_model_class(self):
        """Get the class that is used for :meth:`gensim.models.EnsembleLda.generate_gensim_representation`."""
        if self.topic_model_class is None:
            instruction = (
                'Try setting topic_model_class manually to what the individual models were based on, '
                'e.g. LdaMulticore.'
            )
            try:
                module = importlib.import_module(self.topic_model_module_string)
                self.topic_model_class = getattr(module, self.topic_model_class_string)
                del self.topic_model_module_string
                del self.topic_model_class_string
            except ModuleNotFoundError:
                logger.error(
                    f'Could not import the "{self.topic_model_class_string}" module in order to provide the '
                    f'"{self.topic_model_class_string}" class as "topic_model_class" attribute. {instruction}'
                )
            except AttributeError:
                logger.error(
                    f'Could not import the "{self.topic_model_class_string}" class from the '
                    f'"{self.topic_model_module_string}" module in order to set the "topic_model_class" attribute. '
                    f'{instruction}'
                )
        return self.topic_model_class

    def save(self, *args, **kwargs):
        if self.get_topic_model_class() is not None:
            self.topic_model_module_string = self.topic_model_class.__module__
            self.topic_model_class_string = self.topic_model_class.__name__
        kwargs['ignore'] = frozenset(kwargs.get('ignore', ())).union(('topic_model_class', ))
        super(EnsembleLda, self).save(*args, **kwargs)

    save.__doc__ = SaveLoad.save.__doc__

    def convert_to_memory_friendly(self):
        """Remove the stored gensim models and only keep their ttdas.

        This frees up memory, but you won't have access to the individual models anymore if you intended to use them
        outside of the ensemble.
        """
        self.tms = []
        self.memory_friendly_ttda = True

    def generate_gensim_representation(self):
        """Create a gensim model from the stable topics.

        The returned representation is a Gensim LdaModel (:py:class:`gensim.models.LdaModel`) that has been
        instantiated with an A-priori belief on word probability, eta, that represents the topic-term distribution of
        any stable topics that were found by clustering over the ensemble of topic distributions.

        When no stable topics have been detected, None is returned.

        Returns
        -------
        :py:class:`gensim.models.LdaModel`
            A Gensim LDA Model classic_model_representation for which:
            ``classic_model_representation.get_topics() == self.get_topics()``

        """
        logger.info("generating classic gensim model representations based on results from the ensemble")

        sstats_sum = self.sstats_sum
        # if sstats_sum (which is the number of words actually) should be wrong for some fantastic funny reason
        # that makes you want to peel your skin off, recreate it (takes a while):
        if sstats_sum == 0 and "corpus" in self.gensim_kw_args and not self.gensim_kw_args["corpus"] is None:
            for document in self.gensim_kw_args["corpus"]:
                for token in document:
                    sstats_sum += token[1]
            self.sstats_sum = sstats_sum

        stable_topics = self.get_topics()

        num_stable_topics = len(stable_topics)

        if num_stable_topics == 0:
            logger.error(
                "the model did not detect any stable topic. You can try to adjust epsilon: "
                "recluster(eps=...)"
            )
            self.classic_model_representation = None
            return

        # create a new gensim model
        params = self.gensim_kw_args.copy()
        params["eta"] = self.eta
        params["num_topics"] = num_stable_topics
        # adjust params in a way that no training happens
        params["passes"] = 0  # no training
        # iterations is needed for inference, pass it to the model

        classic_model_representation = self.get_topic_model_class()(**params)

        # when eta was None, use what gensim generates as default eta for the following tasks:
        eta = classic_model_representation.eta
        if sstats_sum == 0:
            sstats_sum = classic_model_representation.state.sstats.sum()
            self.sstats_sum = sstats_sum

        # the following is important for the denormalization
        # to generate the proper sstats for the new gensim model:
        # transform to dimensionality of stable_topics. axis=1 is summed
        eta_sum = 0
        if isinstance(eta, (int, float)):
            eta_sum = [eta * len(stable_topics[0])] * num_stable_topics
        else:
            if len(eta.shape) == 1:  # [e1, e2, e3]
                eta_sum = [[eta.sum()]] * num_stable_topics
            if len(eta.shape) > 1:  # [[e11, e12, ...], [e21, e22, ...], ...]
                eta_sum = np.array(eta.sum(axis=1)[:, None])

        # the factor that will be used when get_topics() is used for normalization
        # will never change because the sum for eta as well as the sum for sstats is constant.
        # Therefore predicting normalization_factor becomes super easy.
        # corpus is a mapping of id to occurrences

        # so one can also easily calculate the
        # right sstats, so that get_topics() will return the stable topics no
        # matter eta.

        normalization_factor = np.array([[sstats_sum / num_stable_topics]] * num_stable_topics) + eta_sum

        sstats = stable_topics * normalization_factor
        sstats -= eta

        classic model_representation.state.sstats = sstats.astype(np.float32)
        # fix expElogbeta.
        classic_model_representation.sync_state()

        self.classic_model_representation = classic_model_representation

        return classic_model_representation

    def add_model(self, target, num_new_models=None):
        """Add the topic term distribution array (ttda) of another model to the ensemble.

        This way, multiple topic models can be connected to an ensemble manually. Make sure that all the models use
        the exact same dictionary/idword mapping.

        In order to generate new stable topics afterwards, use:
            2. ``self.``:meth:`~gensim.models.ensemblelda.EnsembleLda.recluster`

        The ttda of another ensemble can also be used, in that case set ``num_new_models`` to the ``num_models``
        parameter of the ensemble, that means the number of classic models in the ensemble that generated the ttda.
        This is important, because that information is used to estimate "min_samples" for _generate_topic_clusters.

        If you trained this ensemble in the past with a certain Dictionary that you want to reuse for other
        models, you can get it from: ``self.id2word``.

        Parameters
        ----------
        target : {see description}
            1. A single EnsembleLda object
            2. List of EnsembleLda objects
            3. A single Gensim topic model (e.g. (:py:class:`gensim.models.LdaModel`)
            4. List of Gensim topic models

            if memory_friendly_ttda is True, target can also be:
            5. topic-term-distribution-array

            example: [[0.1, 0.1, 0.8], [...], ...]

            [topic1, topic2, ...]
            with topic being an array of probabilities:
            [token1, token2, ...]
