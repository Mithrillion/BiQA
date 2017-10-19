import pandas as pd
import numpy as np
import torch.utils.data as tud
from data_utils import QADataset, get_embeddings
import spacy

lang = 'es'
nlp = spacy.load(lang, vectors=False)
with open("../bilingual_vector/original/wiki.{0}.small.vec".format(lang), "r") as f:
    emb, dic, rev_dic = get_embeddings(f, nr_unk=100, nr_var=600)
print("embedding loaded!")
train = pd.read_pickle("../input_data/train_{0}.pkl".format(lang))
ds = QADataset(train, nlp, rev_dic, True)
print(ds.__len__())
s, q, sl, ql, sv, qv, a = ds.__getitem__(1)
print(s.shape)
print(q)
print(a)
print(sl)
print(ql)
print(sv.shape)
print(qv.shape)

qa_loader = tud.DataLoader(ds, batch_size=32, pin_memory=True, num_workers=3)
# s, q, sl, ql, sv, qv, a = next(iter(qa_loader))
# print(a)
# print(s.shape)
# print(q.shape)
#
for i, qa in enumerate(qa_loader):
    print("batch {0} loaded!".format(i))
