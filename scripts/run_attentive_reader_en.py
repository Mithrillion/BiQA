import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from AttentiveReaderPlus import AttentiveReader
from sklearn.metrics import accuracy_score
from data_utils import QADataset
import shutil
import spacy


seed = 7777
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
hidden_size = 50
n_epochs = 30
var_size = 600
learning_rate = 0.001
resume = False
pack = False


def get_embeddings(vocab, nr_unk=100):
    nr_vector = max(lex.rank for lex in vocab) + 1
    vectors = np.zeros((nr_vector + nr_unk + 2, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector / lex.vector_norm
#             vectors[lex.rank+1] = lex.vector
    return vectors


nlp = spacy.load('en')
emb_vectors = get_embeddings(nlp.vocab, nr_unk=100)

train = pd.read_pickle("../input_data/train_en.pkl")
dev = pd.read_pickle("../input_data/dev_en.pkl")
train_loader = tud.DataLoader(QADataset(train, nlp), batch_size=batch_size, pin_memory=True,
                              num_workers=3, shuffle=True)
dev_loader = tud.DataLoader(QADataset(dev, nlp), batch_size=batch_size, pin_memory=True, num_workers=3)


def save_checkpoint(state, is_best, filename='checkpoint.en.packed.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.en.packed.pth.tar')


def sort_batch(batch, sort_ind=2, pack=True):
    if pack:
        _, orders = torch.sort(batch[sort_ind], dim=0, descending=True)
        return [x[orders] for x in batch]
    else:
        return batch


def validate(net, dev_loader):
    n_dev_batches = len(dev_loader)
    total_val_loss = 0
    outputs = []
    ys = []
    for batch in dev_loader:
        s, q, sl, ql, sm, qm, sv, qv, t = sort_batch(batch, pack=pack)
        s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        sl = Variable(sl, requires_grad=False)
        ql = Variable(ql, requires_grad=False)
        sm = Variable(sm, requires_grad=False)
        qm = Variable(qm, requires_grad=False)
        sv = Variable(sv.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        qv = Variable(qv.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        x = (s, q, sl, ql, sm, qm, sv, qv)
        y = Variable(t.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        out_probability = net.predict(x)

        out_seq = predict_in_domain(batch[6], out_probability)
        val_loss = nn.CrossEntropyLoss()(out_probability, y)
        total_val_loss += val_loss.data.cpu().numpy()[0]
        outputs += [out_seq]
        ys += [t.numpy()]
    loss_score = total_val_loss / n_dev_batches
    combined_outputs = np.concatenate(outputs, 0)
    ys = np.concatenate(ys, 0)
    acc_score = accuracy_score(combined_outputs, ys)
    return loss_score, acc_score


def predict_in_domain(story_vars, out_proba):
    story_vars_array = story_vars.numpy()
    proba_array = out_proba.data.cpu().numpy()
    # find all entities (not non-entity or question placeholder) and get a unique list of their ids
    # var_code - 2 = ent_id (because 0 is non-entity, 1 is @placeholder, 2 is @entity1, etc.)
    ent_vars = [np.unique(np.extract(svar - 2 >= 0, svar - 2)) for svar in story_vars_array]
    out_seq = []
    for i in range(proba_array.shape[0]):
        out_seq.append(ent_vars[i][np.argmax(proba_array[i, ent_vars[i]])])
    return np.array(out_seq)


net = AttentiveReader(var_size, 2000, 50, emb_vectors,
                      dropout=0.2,
                      hidden_size=hidden_size,
                      pack=pack)
net.optimiser = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), learning_rate)

net.cuda()
print("network initialised!")

best_loss = np.inf
if resume:
    print("loading saved states...")
    # resume test
    saved_state = torch.load("./model_best.en.packed.pth.tar")
    net.load_state_dict(saved_state['state_dict'])
    best_loss = saved_state['epoch_val_loss']

print("testing multi-step training!")
for epoch in range(n_epochs):
    print("epoch no.{0}".format(epoch + 1))

    i = 0
    cum_loss = 0
    for i, batch in enumerate(train_loader):

        s, q, sl, ql, sm, qm, sv, qv, y = sort_batch(batch, pack=pack)
        s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        sl = Variable(sl, requires_grad=False)
        ql = Variable(ql, requires_grad=False)
        sm = Variable(sm, requires_grad=False)
        qm = Variable(qm, requires_grad=False)
        sv = Variable(sv.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        qv = Variable(qv.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        y = Variable(y.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        xy = (s, q, sl, ql, sm, qm, sv, qv, y)

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
            print("checkpoint loss = {0:.10}".format(checkpoint_val_loss))
            save_checkpoint({
                'epoch': epoch,
                'batch': i,
                'state_dict': net.state_dict(),
                'epoch_val_loss': checkpoint_val_loss
            }, is_best)
