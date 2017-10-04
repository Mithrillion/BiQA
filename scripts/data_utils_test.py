import pandas as pd
import torch.utils.data as tud
from data_utils import QADataset
import spacy


nlp = spacy.load('es')
train = pd.read_pickle("../input_data/train_es.pkl")
ds = QADataset(train, nlp)
print(ds.__len__())
s, q, sl, ql, sm, qm, sv, qv, a = ds.__getitem__(29521)
print(s.shape)
print(q.shape)
print(a)
print(sl)
print(ql)
print(sm.shape)
print(qm.shape)
print(sv.shape)
print(qv.shape)

qa_loader = tud.DataLoader(ds, batch_size=32, pin_memory=True, num_workers=3)
s, q, sl, ql, sm, qm, sv, qv, a = next(iter(qa_loader))
print(a)
print(s.shape)
print(q.shape)

for i, qa in enumerate(qa_loader):
    print("batch {0} loaded!".format(i))
