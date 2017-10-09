import numpy as np
import pandas as pd
# from gensim.corpora import Dictionary
import spacy
from itertools import chain
import pickle

nlp = spacy.load('es')

dat = pd.read_pickle("../input_data/train_es.pkl")
print(dat.head())
print(dat.shape)

dat = dat.iloc[:1000]  # small

full_corpora = "\n\n".join(list(chain.from_iterable(list(dat['story']) + list(dat['question']))))
print(len(full_corpora))

res = nlp(full_corpora.lower(), parse=False, tag=False, entity=False)
vocab = set([x.text for x in res])
print(len(vocab))
pickle.dump(vocab, open("../input_data/train_vocab_set.pkl", "wb"))

#
# print(vocab_dic)
# vocab_dic.save_as_text("../input_data/train_vocab_dic.txt")
