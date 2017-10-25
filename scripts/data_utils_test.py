import pandas as pd
import numpy as np
import torch.utils.data as tud
from data_utils import QADataset, get_embeddings, BiQADataset
import spacy

lang1 = 'en'
lang2 = 'es'
nlp1 = spacy.load(lang1, vectors=False)
nlp2 = spacy.load(lang2, vectors=False)
with open("../bilingual_vector/original/wiki.{0}.small.vec".format(lang1), "r") as f:
    emb1, dic1, rev_dic1 = get_embeddings(f, nr_unk=100, nr_var=600)
print("embedding loaded!")
with open("../bilingual_vector/original/wiki.{0}.small.vec".format(lang2), "r") as f:
    emb2, dic2, rev_dic2 = get_embeddings(f, nr_unk=100, nr_var=600)
print("embedding loaded!")
train1 = pd.read_pickle("../input_data/train_{0}.pkl".format(lang1))
print(train1.shape)
train2 = pd.read_pickle("../input_data/train_{0}.pkl".format(lang2))
print(train2.shape)
ds = BiQADataset(train1, train2, nlp1, nlp2,  rev_dic1, rev_dic2, True, 5)
print(ds.__len__())
# ln, s, q, sl, ql, sv, qv, a = ds.__getitem__(1)
# print(ln)
# print(s.shape)
# print(q)
# print(a)
# print(sl)
# print(ql)
# print(sv.shape)
# print(qv.shape)
print('-' * 20)

qa_loader = tud.DataLoader(ds, batch_size=32, pin_memory=True, num_workers=3, shuffle=True)
# ln, s, q, sl, ql, sv, qv, a = next(iter(qa_loader))
# print(ln)
# print(a)
# print(s.shape)
# print(q.shape)

for i, qa in enumerate(qa_loader):
    print("batch {0} loaded!".format(i))
