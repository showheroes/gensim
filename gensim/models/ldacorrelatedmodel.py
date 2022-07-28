import logging

import numpy as np
import six
from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.special import gammaln

from gensim import utils
from gensim.matutils import (
    dirichlet_expectation, logsumexp, mean_absolute_difference,
)
from gensim.models.ldamodel import LdaModel, LdaState


logger = logging.getLogger(__name__)


class LdaCorrelatedModel(LdaModel):
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                 iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                 random_state=None, ns_conf=None, minimum_phi_value=0.01,
                 per_word_topics=False, callbacks=None, dtype=np.float32, adaptive_learning_rate=False):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
            If you have a CSC in-memory matrix, you can convert it to a
            streamed corpus with the help of gensim.matutils.Sparse2Corpus.
            If not given, the model is left untrained (presumably because you want to call
            :meth:`~gensim.models.ldamodel.LdaModel.update` manually).
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        distributed : bool, optional
            Whether distributed computing should be used to accelerate training.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        alpha : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on document-topic distribution, this can be:
                * scalar for a symmetric prior over document-topic distribution,
                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,
                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).
        eta : {float, numpy.ndarray of float, list of float, str}, optional
             A-priori belief on topic-word distribution, this can be:
                 * scalar for a symmetric prior over topic-word distribution,
                 * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                 * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.

             Alternatively default prior selecting strategies can be employed by supplying a string:
                 * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                 * 'auto': Learns an asymmetric prior from the corpus.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined.
            Corresponds to :math:`\\kappa` from `'Online Learning for LDA' by Hoffman et al.`_
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
        ns_conf : dict of (str, object), optional
            Key word parameters propagated to :func:`gensim.utils.getNS` to get a Pyro4 nameserver.
            Only used if `distributed` is set to True.
        minimum_phi_value : float, optional
            if `per_word_topics` is True, this represents a lower bound on the term probabilities.
        per_word_topics : bool
            If True, the model also computes a list of topics, sorted in descending order of most likely topics for
            each word, along with their phi values multiplied by the feature length (i.e. word count).
        callbacks : list of :class:`~gensim.models.callbacks.Callback`
            Metric callbacks to log and visualize evaluation metrics of the model during training.
        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional
            Data-type to use during calculations inside model. All inputs are also converted.

        """
        self.dtype = np.finfo(dtype).dtype

        # store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks

        self.adaptive_learning_rate = adaptive_learning_rate

        self.mu_kv = np.zeros((self.num_topics, self.num_terms))
        self.sigma2_kv = np.ones((self.num_topics, self.num_terms))
        self.sigma2_inverse_kv = 1 / self.sigma2_kv

        self.lambda_kv = np.random.uniform(size=(self.num_topics, self.num_terms))
        self.nu2_kv = np.ones((self.num_topics, self.num_terms))
        self.zeta_k = np.sum(np.exp(self.lambda_kv + 0.5 * self.nu2_kv), 1)

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')
        assert self.alpha.shape == (self.num_topics,), \
            "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')
        assert self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms), (
            "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
            (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms))

        self.random_state = utils.get_random_state(random_state)

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # set up distributed environment if necessary
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
            self.numworkers = 1
        else:
            if self.optimize_alpha:
                raise NotImplementedError("auto-optimizing alpha not implemented in distributed LDA")
            # set up distributed version
            try:
                import Pyro4
                if ns_conf is None:
                    ns_conf = {}

                with utils.getNS(**ns_conf) as ns:
                    from gensim.models.lda_dispatcher import LDA_DISPATCHER_PREFIX
                    self.dispatcher = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
                    logger.debug("looking for dispatcher at %s" % str(self.dispatcher._pyroUri))
                    self.dispatcher.initialize(
                        id2word=self.id2word, num_topics=self.num_topics, chunksize=chunksize,
                        alpha=alpha, eta=eta, distributed=False
                    )
                    self.numworkers = len(self.dispatcher.getworkers())
                    logger.info("using distributed version with %i workers", self.numworkers)
            except Exception as err:
                logger.error("failed to initialize distributed LDA (%s)", err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)

        # Initialize the variational distribution q(beta|lambda)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms), dtype=self.dtype)
        self.state.sstats[...] = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(self.get_Elogbeta(self.lambda_kv, self.nu2_kv))

        # Check that we haven't accidentally fallen back to np.float64
        assert self.eta.dtype == self.dtype
        assert self.expElogbeta.dtype == self.dtype

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, chunks_as_numpy=use_numpy)

    def sync_state(self, current_Elogbeta=None):
        """Propagate the states topic probabilities to the inner object's attribute.

        Parameters
        ----------
        current_Elogbeta: numpy.ndarray
            Posterior probabilities for each topic, optional.
            If omitted, it will get Elogbeta from state.

        """
        if current_Elogbeta is None:
            current_Elogbeta = self.get_Elogbeta(self.lambda_kv, self.nu2_kv)
        self.expElogbeta = np.exp(current_Elogbeta)
        assert self.expElogbeta.dtype == self.dtype

    def do_mstep(self, rho, other, extra_pass=False):
        """Maximization step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.

        Parameters
        ----------
        rho : float
            Learning rate.
        other : :class:`~gensim.models.ldamodel.LdaModel`
            The model whose sufficient statistics will be used to update the topics.
        extra_pass : bool, optional
            Whether this step required an additional pass over the corpus.

        """
        logger.debug("updating topics")
        # update self with the new blend; also keep track of how much did
        # the topics change through this update, to assess convergence
        previous_Elogbeta = self.get_Elogbeta(self.lambda_kv, self.nu2_kv)
        self.state.blend(rho, other)

        for k in range(self.num_topics):
            self.optimize_zeta_for_topic(k)
            self.optimize_lambda_for_topic(k)
            self.optimize_zeta_for_topic(k)
            self.optimize_nu2_for_topic(k)
            self.optimize_zeta_for_topic(k)

        current_Elogbeta = self.get_Elogbeta(self.lambda_kv, self.nu2_kv)
        self.sync_state(current_Elogbeta)

        # print out some debug info at the end of each EM iteration
        self.print_topics(5)
        diff = mean_absolute_difference(previous_Elogbeta.ravel(), current_Elogbeta.ravel())
        logger.info("topic diff=%f, rho=%f", diff, rho)

        if self.optimize_eta:
             self.update_eta(self.state.get_lambda(), rho)

        if not extra_pass:
            # only update if this isn't an additional pass
            self.num_updates += other.numdocs

    def optimize_zeta_for_topic(self, k):
        self.zeta_k[k] = np.sum(np.exp(self.lambda_kv[k] + 0.5 * self.nu2_kv[k]))

    def optimize_lambda_for_topic(self, k):
        def f(lambda_v, k):
            result = 0.0
            result -= 0.5 * np.sum((lambda_v - self.mu_kv[k]) * self.sigma2_inverse_kv[k] * (lambda_v - self.mu_kv[k]))
            result += np.sum(self.state.sstats[k] * (lambda_v - (1/self.zeta_k[k]) * np.exp(lambda_v + 0.5 * self.nu2_kv[k])))

            return -result

        def df(lambda_v, k):
            result_v = np.zeros(self.num_terms)
            result_v -= self.sigma2_inverse_kv[k] * (lambda_v - self.mu_kv[k])
            result_v += self.state.sstats[k] * (1 - (1/self.zeta_k[k]) * np.exp(lambda_v + 0.5 * self.nu2_kv[k]))

            return -result_v

        res = minimize(f, self.lambda_kv[k], args=(k,), method="BFGS", jac=df)
        assert res.success == True
        self.lambda_kv[k] = res.x

    def optimize_nu2_for_topic(self, k):
        def f(nu2_v, k):
            result = 0.0
            result -= 0.5 * np.sum(nu2_v * self.sigma2_inverse_kv[k])
            result -= np.sum(self.state.sstats[k] * (1/self.zeta_k[k]) * np.exp(self.lambda_kv[k] + 0.5 * nu2_v))
            result += 0.5 * np.sum(np.log(nu2_v) + np.log(2 * np.pi) + 1)

            return -result

        def df(nu2_v, k):
            result_v = np.zeros(self.num_terms)
            result_v -= 0.5 * self.sigma2_inverse_kv[k]
            result_v -= (self.state.sstats[k] / (2 * self.zeta_k[k])) * np.exp(self.lambda_kv[k] + 0.5 * nu2_v)
            result_v += 1 / (2 * nu2_v)

            return -result_v

        # we use epsilon as min to prevent division by 0 and log from 0
        epsilon = np.finfo(self.dtype).eps
        constraints = [(epsilon, None) for _ in range(self.num_terms)]

        res = fmin_l_bfgs_b(f, self.nu2_kv[k], args=(k,), fprime=df, bounds=constraints)
        assert res[2]["warnflag"] == 0
        self.nu2_kv[k] = res[0]

    def get_Elogbeta(self, lambda_kv, nu2_kv):
        zeta_k1 = self.zeta_k[:, np.newaxis]
        Elogbeta = np.zeros((self.num_topics, self.num_terms), dtype=self.dtype)
        Elogbeta += lambda_kv - np.log(zeta_k1) + 1
        Elogbeta -= (1/zeta_k1) * np.sum(np.exp(lambda_kv + 0.5 * nu2_kv), 1, keepdims=True)

        return Elogbeta

    def bound(self, corpus, gamma=None, lambda_kv=None, nu2_kv=None, subsample_ratio=1.0):
        """Estimate the variational bound of documents from the corpus as E_q[log p(corpus)] - E_q[log q(corpus)].

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to estimate the
            variational bounds.
        gamma : numpy.ndarray, optional
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lambda_kv : numpy.ndarray, optional
            Variational parameter lambda of words-topic distribution. If not supplied, it will take current value.
        nu2_kv : numpy.ndarray, optional
            Variational parameter nu2 of words-topic distribution. If not supplied, it will take current value.
        subsample_ratio : float, optional
            Percentage of the whole corpus represented by the passed `corpus` argument (in case this was a sample).
            Set to 1.0 if the whole corpus was passed.This is used as a multiplicative factor to scale the likelihood
            appropriately.

        Returns
        -------
        numpy.ndarray
            The variational bound score calculated for each document.

        """
        if lambda_kv is None:
            lambda_kv = self.lambda_kv

        if nu2_kv is None:
            nu2_kv = self.nu2_kv

        score = 0.0
        Elogbeta = self.get_Elogbeta(lambda_kv, nu2_kv)

        for d, doc in enumerate(corpus):  # stream the input doc-by-doc, in case it's too large to fit in RAM
            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)

            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype

            # E[log p(doc | theta, beta)]
            score += sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)

            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))

        # Compensate likelihood for when `corpus` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always roughly on the same scale.
        score *= subsample_ratio

        # E[log p(beta | mu, Sigma)]
        Elogpbeta_k = np.zeros(self.num_topics)
        Elogpbeta_k += 0.5 * np.log(np.prod(self.sigma2_inverse_kv, 1))
        Elogpbeta_k -= 0.5 * np.sum(nu2_kv * self.sigma2_inverse_kv, 1)
        Elogpbeta_k -= 0.5 * np.sum((lambda_kv - self.mu_kv) * self.sigma2_inverse_kv * (lambda_kv - self.mu_kv), 1)

        # E[log q(beta | lambda, nu2)]
        Elogqbeta_k = np.zeros(self.num_topics)
        Elogqbeta_k -= 0.5 * (np.sum(np.log(nu2_kv), 1) + self.num_terms)

        score += np.sum(Elogpbeta_k - Elogqbeta_k)

        return score
