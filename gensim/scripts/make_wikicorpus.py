#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX [VOCABULARY_SIZE]

Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This actually creates several files:

* `OUTPUT_PREFIX_wordids.txt.bz2`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation in Matrix Market format
* `OUTPUT_PREFIX_bow.mm.index`: index for `OUTPUT_PREFIX_bow.mm`
* `OUTPUT_PREFIX_bow.mm.metadata.cpickle`: titles of documents
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation in Matix Market format
* `OUTPUT_PREFIX_tfidf.mm.index`: index for `OUTPUT_PREFIX_tfidf.mm`
* `OUTPUT_PREFIX.tfidf_model`: TF-IDF model

The output Matrix Market files can then be compressed (e.g., by bzip2) to save
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing tokens that appear in more than 10%% of all documents). Defaults to
100,000.

If you have the `pattern` package installed, this script will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern .

Example:
  python -m gensim.scripts.make_wikicorpus ~/gensim/results/enwiki-latest-pages-articles.xml.bz2 ~/gensim/results/wiki
"""


import logging
import os.path
import sys

from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel
import gensim.utils as gu
from adhero.core.nlp.utils import Stopwords

# Wiki is first scanned for all distinct word types (~7M). The types that
# appear in more than 10% of articles are removed and from the rest, the
# DEFAULT_DICT_SIZE most frequent types are kept.
DEFAULT_DICT_SIZE = 100000


def my_lemmatize(content, min_len, max_len, language):
    language_codes = {
        'german' : 'de',
        'english' : 'en',
        'french' : 'fr',
        'dutch' : 'nl',
        'russian' : 'ru'
    }
    lc = language_codes[language]
    #special case english
    if lc == 'en':
        nlp = space.load('en_core_web_sm')
    else:
        nlp = spacy.load(f'{lc}_core_news_sm')

    sw_class = Stopwords()

    stopwords = sw_class[language]
    # lang_class = spu.get_lang_class(language_codes[language])
    # lang = lang_class()
    # lemmatizer = lang.Defaults.create_lemmatizer()

    doc = nlp(content)
    # content = gu.tokenize(content, lower=True, errors='ignore')
    ret_val = []
    # lemmas = list(map(lemmatizer, content))
    non_words = re.compile('[^\W]')
    for token in doc:
        l = token.lemma_.strip()
        l = non_words.sub('', l)
        l = re.sub(r'\s+', ' ', l)
        l = l.strip()
        if not l:
            continue
        l = l.lower()
        if min_len <= len(l) <= max_len and token.text not in stopwords and l not in stopwords and not token.is_stop:
            ret_val.append(l)

    return ret_val


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    if not os.path.isdir(os.path.dirname(outp)):
        raise SystemExit("Error: The output directory does not exist. Create the directory and try again.")

    if len(sys.argv) > 3:
        keep_words = int(sys.argv[3])
    else:
        keep_words = DEFAULT_DICT_SIZE

    language = sys.argv[4]
    online = 'online' in program
    lemmatize = 'lemma' in program
    debug = 'nodebug' not in program



    if online:
        dictionary = HashDictionary(id_range=keep_words, debug=debug)
        dictionary.allow_update = True  # start collecting document frequencies
        wiki = WikiCorpus(inp, lemmatize=lemmatize, dictionary=dictionary)
        # ~4h on my macbook pro without lemmatization, 3.1m articles (august 2012)
        MmCorpus.serialize(outp + '_bow.mm', wiki, progress_cnt=10000, metadata=True)
        # with HashDictionary, the token->id mapping is only fully instantiated now, after `serialize`
        dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)
        dictionary.save_as_text(outp + '_wordids.txt.bz2')
        wiki.save(outp + '_corpus.pkl.bz2')
        dictionary.allow_update = False
    else:
        if lemmatize:
            wiki = WikiCorpus(inp, lemmatizer_func=my_lemmatize)  # takes about 9h on a macbook pro, for 3.5m articles (june 2011)
        else:
            wiki = WikiCorpus(inp)  # takes about 9h on a macbook pro, for 3.5m articles (june 2011)
        # only keep the most frequent words (out of total ~8.2m unique tokens)
        wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)
        # save dictionary and bag-of-words (term-document frequency matrix)
        MmCorpus.serialize(outp + '_bow.mm', wiki, progress_cnt=10000, metadata=True)  # another ~9h
        wiki.dictionary.save_as_text(outp + '_wordids.txt.bz2')
        # load back the id->word mapping directly from file
        # this seems to save more memory, compared to keeping the wiki.dictionary object from above
        dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')
    del wiki

    # initialize corpus reader and word->id mapping
    mm = MmCorpus(outp + '_bow.mm')

    # build tfidf, ~50min
    tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
    tfidf.save(outp + '.tfidf_model')

    # save tfidf vectors in matrix market format
    # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
    MmCorpus.serialize(outp + '_tfidf.mm', tfidf[mm], progress_cnt=10000)

    logger.info("finished running %s", program)
