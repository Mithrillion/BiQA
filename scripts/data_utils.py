import numpy as np
import re
import torch.utils.data as tud


emb_dim = 50


def get_word_ids(doc, rnn_encode=False, max_length=100, nr_unk=100):
    queue = list(doc)
    X = np.zeros(max_length, dtype='int32')
    M = np.zeros(max_length, dtype='int32')
    V = np.zeros(max_length, dtype='int32')
    words = []
    while len(words) <= max_length and queue:
        word = queue.pop(0)
        if rnn_encode or (not word.is_punct and not word.is_space):
            words.append(word)
    words.sort()
    for j, token in enumerate(words):
        if token.text == '@placeholder':
            V[j] = 1
        elif token.text[:7] == '@entity':
            # temporary dix
            # TODO: properly fix entity replacement
            num = int(re.search(r'\d+', token.text[7:]).group(0))
            if 0 <= num < 500:
                V[j] = num + 2
        if token.has_vector:
            X[j] = token.rank + 1
            M[j] = 1
        else:
            X[j] = (token.shape % (nr_unk - 1)) + 2
        j += 1
        if j >= max_length:
            break
    return X, M, V


class QADataset(tud.Dataset):
    def __init__(self, data_df, nlp):
        self.data_df = data_df
        self.nlp = nlp

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, i):

        story = self.nlp(self.data_df['story'].iloc[i], parse=False, tag=False, entity=False)
        s, s_mask, s_var = get_word_ids(story, max_length=2000)
        s_len = np.sum(s != 0)

        question = self.nlp(self.data_df['question'].iloc[i], parse=False, tag=False, entity=False)
        q, q_mask, q_var = get_word_ids(question, max_length=50)
        q_len = np.sum(q != 0)

        answer = int(re.search(r'\d+', self.data_df['answer'].iloc[i]).group(0))

        # # TODO: REMOVE DEBUG!
        # if q_len <= 0:
        #     raise RuntimeError('Question zero length!, ID={0}'.format(i))

        return s, q, s_len, q_len, s_mask, q_mask, s_var, q_var, answer
