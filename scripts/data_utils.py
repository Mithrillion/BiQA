import numpy as np
import re
import torch.utils.data as tud
import torch
import shutil


def get_word_ids(doc, rnn_encode=False, max_length=100, nr_unk=100, nr_var=600, dic=None):
    dic = {v: k for k, v in dic.items()}
    queue = list(doc)
    X = np.zeros(max_length, dtype='int32')
    # M = np.zeros(max_length, dtype='int32')
    V = np.zeros(max_length, dtype='int32')
    words = []
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
            num = int(re.search(r'\d+', token.text[7:]).group(0))
            if 0 <= num < nr_var:
                X[j] = num + 2
                V[j] = num + 2
        elif token.has_vector:
            X[j] = dic[token.rank] + nr_unk + nr_var + 2
            # M[j] = 1
        else:
            # X: [null; ph; vars; unks; vocab]
            X[j] = (token.shape % nr_unk) + 2 + nr_var
        if j >= max_length - 1:
            break
    return X, V


class QADataset(tud.Dataset):
    def __init__(self, data_df, nlp, dic):
        self.data_df = data_df
        self.nlp = nlp
        self.dic = dic

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, i):

        story = self.nlp(self.data_df['story'].iloc[i].lower(), parse=False, tag=False, entity=False)
        s, s_var = get_word_ids(story, max_length=2000, dic=self.dic)
        s_len = np.sum(s != 0)

        question = self.nlp(self.data_df['question'].iloc[i].lower(), parse=False, tag=False, entity=False)
        q, q_var = get_word_ids(question, max_length=50, dic=self.dic)
        q_len = np.sum(q != 0)

        answer = int(re.search(r'\d+', self.data_df['answer'].iloc[i]).group(0))

        # # TODO: REMOVE DEBUG!
        # if q_len <= 0:
        #     raise RuntimeError('Question zero length!, ID={0}'.format(i))

        return s, q, s_len, q_len, s_var, q_var, answer


def get_embeddings(vocab, nr_unk=100, nr_var=600):
    # nr_vector = max(lex.rank for lex in vocab) + 1
    nr_vector = np.sum([x.has_vector for x in vocab])
    vectors = np.zeros((nr_vector + nr_unk + nr_var + 2, vocab.vectors_length), dtype='float32')
    dic = dict()
    i = 0
    for lex in vocab:
        if lex.has_vector:
            vectors[i + nr_unk + nr_var + 2] = lex.vector / lex.vector_norm
            dic[i] = lex.rank
            i += 1
    return vectors, dic


def save_checkpoint(state, is_best, filename='checkpoint.en.packed.pth.tar',
                    best_name='model_best.en.packed.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


def sort_batch(batch, sort_ind=2, pack=True):
    if pack:
        _, orders = torch.sort(batch[sort_ind], dim=0, descending=True)
        return [x[orders] for x in batch]
    else:
        return batch
