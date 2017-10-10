import numpy as np
import pandas as pd
import spacy
import re
import torch.utils.data as tud
from data_utils import get_word_ids, QADataset, save_checkpoint, sort_batch

lang = 'es'
nr_unk = 100
nr_var = 600

train = pd.read_pickle("../input_data/train_{0}.pkl".format(lang))
dev = pd.read_pickle("../input_data/dev_{0}.pkl".format(lang))
test = pd.read_pickle("../input_data/test_{0}.pkl".format(lang))

combined = pd.concat([train, dev, test], axis=0)


def get_embeddings(f, nr_unk=100, nr_var=600):
    nr_vector, ndim = f.readline().split(" ")
    # nr_vector = int(nr_vector)
    ndim = int(ndim)
    # vectors = np.zeros((nr_vector + nr_unk + nr_var + 2, ndim), dtype='float32')
    dic = dict()
    i = 0
    line = f.readline()
    while line:
        parts = line.split(" ")
        if len(parts) != ndim + 2:
            print(line)
            raise ValueError("Vector size mismatch!")
        else:
            word = parts[0]
            # vec = np.array(parts[1:-1]).astype(np.float32)
            # vectors[i + nr_unk + nr_var + 2, :] = vec / np.linalg.norm(vec)
            dic[i] = word
            i += 1
        line = f.readline()
    rev_dic = {v: k for k, v, in dic.items()}
    return ndim, dic, rev_dic


nlp = spacy.load(lang, vectors=False)
# load new vectors
with open("../wordvecs/wiki.{0}/wiki.{0}.nospace.vec".format(lang), "r") as f:
    ndim, dic, rev_dic = get_embeddings(f, nr_unk=100, nr_var=600)

ds = QADataset(combined, nlp, rev_dic)
qa_loader = tud.DataLoader(ds, batch_size=32, pin_memory=True, num_workers=3)

vocab_set = set()
for i, qa in enumerate(qa_loader):
    if i % 1000 == 0:
        print("current batch {0}/{1}".format(i, len(qa_loader)))
    s, q, _, _, _, _, _ = qa
    cur_set = set(np.unique(s.numpy())).union(set(np.unique(q.numpy())))
    vocab_set = vocab_set.union(cur_set)

print(len(vocab_set))
valid = set([x - (nr_unk + nr_var + 2) for x in vocab_set if x >= nr_unk + nr_var + 2])

with open("../wordvecs/wiki.{0}/wiki.{0}.vec".format(lang)) as f:
    with open("../wordvecs/wiki.{0}/wiki.{0}.small.vec".format(lang), "w") as g:
        g.write("{0} {1}\n".format(len(vocab_set), ndim))
        f.readline()
        i = 0
        line = f.readline()
        while line:
            if not re.search(r'[\u00A0\u1680\u180e\u2000-\u2009\u200a\u200b\u202f\u205f\u3000\u2028\x85]',
                             line):
                word = line.split(" ")[0]
                if i in valid:
                    g.write(line)
                    i += 1
            line = f.readline()
