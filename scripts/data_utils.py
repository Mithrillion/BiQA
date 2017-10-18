import numpy as np
import re
import torch.utils.data as tud
import torch
import shutil


def get_word_ids(doc, rnn_encode=True, max_length=100,
                 nr_unk=100, nr_var=600, rev_dic=None, relabel=True, ent_dict=None):
    queue = list(doc)
    X = np.zeros(max_length, dtype='int32')
    # M = np.zeros(max_length, dtype='int32')
    V = np.zeros(max_length, dtype='int32')
    words = []
    if ent_dict is None:
        ent_dict = {}
    k = 0
    while len(words) <= max_length and queue:
        word = queue.pop(0)
        if rnn_encode or (not word.is_punct and not word.is_space):
            words.append(word)
    words.sort()
    for j, token in enumerate(words):
        if token.text == '@placeholder':
            X[j] = 1
            V[j] = 1
        elif token.text[:7] == '@entity':
            # temporary dix
            # TODO: properly fix entity replacement
            try:
                num = int(re.search(r'\d+', token.text[7:]).group(0))
                if 0 <= num < nr_var:
                    if relabel:
                        if num not in ent_dict.keys():
                            ent_dict[num] = k
                            k += 1
                        X[j] = ent_dict[num] + 2
                        V[j] = ent_dict[num] + 2
                    else:
                        X[j] = num + 2
                        V[j] = num + 2
            except AttributeError:
                X[j] = (token.shape % nr_unk) + 2 + nr_var
        elif token.text in rev_dic.keys():
            X[j] = rev_dic[token.text] + nr_unk + nr_var + 2
            # M[j] = 1
        else:
            # X: [null; ph; vars; unks; vocab]
            X[j] = (token.shape % nr_unk) + 2 + nr_var
        if j >= max_length - 1:
            break
    return X, V, ent_dict


class QADataset(tud.Dataset):
    def __init__(self, data_df, nlp, rev_dic, relabel=True):
        self.data_df = data_df
        self.nlp = nlp
        self.rev_dic = rev_dic
        self.relabel = relabel

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, i):

        story = self.nlp(self.data_df['story'].iloc[i].lower(), parse=False, tag=False, entity=False)
        s, s_var, ent_dict = get_word_ids(story, max_length=2000, rev_dic=self.rev_dic, relabel=self.relabel)
        s_len = np.sum(s != 0)

        question = self.nlp(self.data_df['question'].iloc[i].lower(), parse=False, tag=False, entity=False)
        q, q_var, ent_dict = get_word_ids(question, max_length=50, rev_dic=self.rev_dic, relabel=self.relabel,
                                          ent_dict=ent_dict)
        q_len = np.sum(q != 0)

        if self.relabel:
            answer = ent_dict[int(re.search(r'\d+', self.data_df['answer'].iloc[i]).group(0))]
        else:
            answer = int(re.search(r'\d+', self.data_df['answer'].iloc[i]).group(0))

        return s, q, s_len, q_len, s_var, q_var, answer


# def get_embeddings(vocab, nr_unk=100, nr_var=600):
#     # nr_vector = max(lex.rank for lex in vocab) + 1
#     nr_vector = np.sum([x.has_vector for x in vocab])
#     vectors = np.zeros((nr_vector + nr_unk + nr_var + 2, vocab.vectors_length), dtype='float32')
#     dic = dict()
#     i = 0
#     np.seterr(all='raise')
#     for lex in vocab:
#         if lex.has_vector:
#             try:
#                 vectors[i + nr_unk + nr_var + 2, :] = lex.vector / lex.vector_norm
#             except FloatingPointError:
#                 # TODO: SpaCy bug that introduces new erroneous vectors that should not exist
#                 pass
#             dic[i] = lex.rank
#             i += 1
#     return vectors, dic


def get_embeddings(f, nr_unk=100, nr_var=600, meta=None):
    if meta is None:
        nr_vector, ndim = f.readline().split(" ")
    else:
        nr_vector, ndim = meta.split(" ")
    nr_vector = int(nr_vector)
    ndim = int(ndim)
    vectors = np.zeros((nr_vector + nr_unk + nr_var + 2, ndim), dtype='float32')
    dic = dict()
    i = 0
    line = f.readline()
    while line:
        parts = line.split(" ")
        if len(parts) != ndim + 1 and len(parts) != ndim + 2:
            print(line)
            raise ValueError("Vector size mismatch! Got {0}, expected {1} (+1)!".
                             format(len(parts), ndim + 1))
        else:
            word = parts[0]
            vec = np.array(parts[1: 1 + ndim]).astype(np.float32)
            vectors[i + nr_unk + nr_var + 2, :] = vec / np.linalg.norm(vec)
            dic[i] = word
            i += 1
        line = f.readline()
    rev_dic = {v: k for k, v, in dic.items()}
    return vectors, dic, rev_dic


def save_checkpoint(state, is_best, filename='checkpoint.en.packed.pth.tar',
                    best_name='model_best.en.packed.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


def sort_batch(batch, sort_ind=3, pack=True):
    if pack:
        _, orders = torch.sort(batch[sort_ind], dim=0, descending=True)
        return [x[orders] for x in batch]
    else:
        return batch

