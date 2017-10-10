import numpy as np
import re
import pandas as pd
import spacy
from data_utils import get_embeddings, get_word_ids


lang = 'es'
nlp = spacy.load(lang, vectors=False)
with open("../wordvecs/wiki.{0}/wiki.{0}.small.vec".format(lang), "r") as f:
    nlp.vocab.load_vectors(f)
# load new vectors
with open("../bilingual_vector/original/wiki.{0}.small.vec".format(lang), "r") as f:
    emb, dic, rev_dic = get_embeddings(f, nr_unk=100, nr_var=600)
print("embedding loaded!")
dat = pd.read_pickle("../input_data/dev_{0}.pkl".format(lang))
temp = nlp(dat['question'].iloc[0].lower())
print(temp)
print(temp[2])
print(temp[2].vector, temp[2].vector.shape)
print(emb.shape)
X, V = get_word_ids(temp, max_length=50, rev_dic=rev_dic)
print(list(X))
print(list(V))
print(emb[X[2], :] / (temp[2].vector / temp[2].vector_norm))
print()