import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from AttentiveReaderPlus import AttentiveReader
from sklearn.metrics import accuracy_score
from data_utils import QADataset, get_embeddings, save_checkpoint, sort_batch
import spacy


seed = 7777
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
hidden_size = 128
n_epochs = 30
var_size = 600
dropout = 0.2
learning_rate = 0.001
story_rec_layers = 1
resume = False
pack = True
emb_trainable = False
lang = 'en'
# use original


nlp = spacy.load(lang, vectors=False)
# load new vectors
# with open("../rc_data/processed/glove.6B/glove.6B.100d.txt".format(lang), "r") as f:
with open("../bilingual_vector/original/wiki.{0}.small.vec".format(lang), "r") as f:
    emb_vectors, dic, rev_dic = get_embeddings(f, nr_unk=100, nr_var=600)
print("embedding loaded!")

train = pd.read_pickle("../input_data/train_{0}.pkl".format(lang))
dev = pd.read_pickle("../input_data/dev_{0}.pkl".format(lang))
train_loader = tud.DataLoader(QADataset(train, nlp, rev_dic), batch_size=batch_size, pin_memory=True,
                              num_workers=3, shuffle=True)
dev_loader = tud.DataLoader(QADataset(dev, nlp, rev_dic), batch_size=batch_size, pin_memory=True, num_workers=3)


def validate(net, dev_loader):
    n_dev_batches = len(dev_loader)
    total_val_loss = 0
    outputs = []
    ys = []
    for batch in dev_loader:
        s, q, sl, ql, sv, qv, t = sort_batch(batch, pack=pack, sort_ind=3)
        s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        sl = Variable(sl, requires_grad=False)
        ql = Variable(ql, requires_grad=False)
        x = (s, q, sl, ql)
        y = Variable(t.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        out_logits = net.forward(x)

        out_seq = predict_in_domain(batch[4], out_logits)
        val_loss = nn.CrossEntropyLoss()(out_logits, y)
        total_val_loss += val_loss.data.cpu().numpy()[0]
        outputs += [out_seq]
        ys += [t.numpy()]
    loss_score = total_val_loss / n_dev_batches
    combined_outputs = np.concatenate(outputs, 0)
    ys = np.concatenate(ys, 0)
    acc_score = accuracy_score(combined_outputs, ys)
    return loss_score, acc_score


def predict_in_domain(story_vars, out_logits):
    story_vars_array = story_vars.numpy()
    logit_array = out_logits.data.cpu().numpy()
    # find all entities (not non-entity or question placeholder) and get a unique list of their ids
    # var_code - 2 = ent_id (because 0 is non-entity, 1 is @placeholder, 2 is @entity1, etc.)
    ent_vars = [np.unique(np.extract(svar - 2 >= 0, svar - 2)) for svar in story_vars_array]
    out_seq = []
    for i in range(logit_array.shape[0]):
        out_seq.append(ent_vars[i][np.argmax(logit_array[i, ent_vars[i]])])
    return np.array(out_seq)


net = AttentiveReader(var_size, 2000, 50, emb_vectors,
                      dropout=dropout,
                      hidden_size=hidden_size,
                      pack=pack,
                      emb_trainable=emb_trainable,
                      story_rec_layers=story_rec_layers)
net.optimiser = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), learning_rate)
# net.optimiser = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

net.cuda()
print("network initialised!")

best_loss = np.inf
init_epoch = 0
if resume:
    print("loading saved states...")
    # resume test
    saved_state = torch.load("./checkpoint.{0}.packed.pth.tar".format(lang))
    best_state = torch.load("./model_best.{0}.packed.pth.tar".format(lang))
    net.load_state_dict(saved_state['state_dict'])
    init_epoch = saved_state['epoch']
    best_loss = best_state['epoch_val_loss']
    del best_state

print("testing multi-step training!")
for epoch in range(init_epoch, n_epochs):
    print("epoch no.{0}".format(epoch + 1))

    i = 0
    cum_loss = 0
    for i, batch in enumerate(train_loader):

        s, q, sl, ql, sv, qv, y = sort_batch(batch, pack=pack, sort_ind=3)
        s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        sl = Variable(sl, requires_grad=False)
        ql = Variable(ql, requires_grad=False)
        y = Variable(y.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        xy = (s, q, sl, ql, y)

        net.train()
        loss, out = net.train_on_batch(xy)
        cum_loss += loss.cpu().numpy()[0]

        if i % 100 == 1:
            net.eval()
            if i == 1:
                cum_loss *= 50
            print("iteration {0}/{1}\ntraining loss =   {2:.10}".
                  format(i, len(train_loader), cum_loss / 100.))
            cum_loss = 0
        if i % 1000 == 1 or i == len(train_loader) - 1:
            net.eval()
            val_loss, acc = validate(net, dev_loader)
            print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                  format(val_loss, acc))
            checkpoint_val_loss = val_loss
            is_best = checkpoint_val_loss < best_loss
            if is_best:
                best_loss = checkpoint_val_loss
            # print("checkpoint loss = {0:.10}".format(checkpoint_val_loss))
            save_checkpoint(
                {
                    'epoch': epoch,
                    'batch': i,
                    'state_dict': net.state_dict(),
                    'epoch_val_loss': checkpoint_val_loss
                },
                is_best,
                filename='checkpoint.{0}.packed.pth.tar'.format(lang),
                best_name='model_best.{0}.packed.pth.tar'.format(lang))
