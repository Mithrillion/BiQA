import pandas as pd
from multiprocessing import Pool
import re


train = pd.read_hdf("../rc_data/combined.h5", "cnn_train")
dev = pd.read_hdf("../rc_data/combined.h5", "cnn_dev")
test = pd.read_hdf("../rc_data/combined.h5", "cnn_test")
train.rename(columns={"query": "question"}, inplace=True)
dev.rename(columns={"query": "question"}, inplace=True)
test.rename(columns={"query": "question"}, inplace=True)

def ent_num(x):
    try:
        return int(re.search(r'\d+', x).group(0))
    except AttributeError:
        print(x)
        return None


with Pool(4) as p:
    res = p.map(ent_num, train['answer'])

train.loc[~pd.isnull(res)].to_pickle("../input_data/train_en.pkl")


with Pool(4) as p:
    res = p.map(ent_num, dev['answer'])

dev.loc[~pd.isnull(res)].to_pickle("../input_data/dev_en.pkl")


with Pool(4) as p:
    res = p.map(ent_num, test['answer'])

test.loc[~pd.isnull(res)].to_pickle("../input_data/test_en.pkl")
