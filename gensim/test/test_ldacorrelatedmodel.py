import logging
import unittest

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldacorrelatedmodel
from gensim.test.test_ldamodel import TestLdaModel
from gensim.test.utils import datapath, common_texts

dictionary = Dictionary(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]


class TestLdaCorrelatedModel(TestLdaModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldacorrelatedmodel.LdaCorrelatedModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
