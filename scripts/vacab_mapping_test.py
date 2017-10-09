import numpy as np
import re
import pandas as pd
import spacy


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


def get_embeddings(vocab, nr_unk=100, nr_var=600):
    # nr_vector = max(lex.rank for lex in vocab) + 1
    nr_vector = np.sum([x.has_vector for x in nlp.vocab])
    vectors = np.zeros((nr_vector + nr_unk + nr_var + 2, vocab.vectors_length), dtype='float32')
    dic = dict()
    i = 0
    for lex in vocab:
        if lex.has_vector:
            vectors[i + nr_unk + nr_var + 2] = lex.vector / lex.vector_norm
            dic[i] = lex.rank
            i += 1
    return vectors, dic


nlp = spacy.load('en')
with open("../wordvecs/wiki.en/wiki.en.small.vec", "r") as f:
    nlp.vocab.load_vectors(f)
emb, dic = get_embeddings(nlp.vocab)
dat = pd.read_pickle("../input_data/dev_en.pkl")
temp = nlp(dat['question'][1].lower())
print(temp)
print(temp[2])
print(temp[2].vector, temp[2].vector.shape)
print(emb.shape)
X, V = get_word_ids(temp, max_length=50, dic=dic)
print(list(X))
print(list(V))
print(emb[X[2], :] / (temp[2].vector / temp[2].vector_norm))
print()
