import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from AttentiveReader import AttentiveReader
from sklearn.metrics import accuracy_score
import pickle
import shutil

# # insert this to the top of your scripts (usually main.py)
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#     traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True

seed = 7777
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
n_epochs = 30
use_cuda = True
resume = True
pack = True

stories = np.load("../rc_data/train_stories.npy").astype(np.long)
queries = np.load("../rc_data/train_queries.npy").astype(np.long)
answers = np.load("../rc_data/train_answers.npy").astype(np.long)

dev_stories = np.load("../rc_data/dev_stories.npy").astype(np.long)
dev_queries = np.load("../rc_data/dev_queries.npy").astype(np.long)
dev_answers = np.load("../rc_data/dev_answers.npy").astype(np.long)

full_answers = pickle.load(open("../rc_data/answer_vocab.pkl", "rb"))
answers_set = set(full_answers)


def save_checkpoint(state, is_best, filename='checkpoint.packed.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.packed.pth.tar')


def validate(net, dso, dqo, dao, ds_len, dq_len, dev_batches, cuda=True, pack=True):
    n_dev_batches = len(dev_batches)
    total_val_loss = 0
    outputs = []
    for j in range(n_dev_batches):
        start, end = dev_batches[j]
        if cuda:
            ds = Variable(dso[start:end].cuda(async=True), volatile=True)
            dq = Variable(dqo[start:end].cuda(async=True), volatile=True)
            da = Variable(dao[start:end].cuda(async=True), volatile=True)
        else:
            ds = Variable(dso[start:end], volatile=True)
            dq = Variable(dqo[start:end], volatile=True)
            da = Variable(dao[start:end], volatile=True)
        if pack:
            out_probability = net.predict(ds, dq, ds_len[start:end], dq_len[start:end], pack)
        else:
            out_probability = net.predict(ds, dq, None, None, pack)
        # _, out_seq = torch.max(out_probability, 1)
        # out_seq = out_seq.data.cpu().numpy()
        out_seq = predict_in_domain(ds, out_probability)
        val_loss = nn.CrossEntropyLoss()(out_probability, da)
        total_val_loss += val_loss
        outputs += [out_seq]
    loss_score = total_val_loss / n_dev_batches
    combined_outputs = np.concatenate(outputs, 0)
    acc_score = accuracy_score(combined_outputs, answers_code_dev)
    return loss_score, acc_score


def predict_in_domain(stories, out_proba, answers_set=answers_set):
    story_array = stories.data.cpu().numpy()
    proba_array = out_proba.data.cpu().numpy()
    answers = [[entity2code[x] for x in set(words).intersection(answers_set)] for words in story_array]
    out_seq = []
    for i in range(proba_array.shape[0]):
        out_seq += [answers[i][np.argmax(proba_array[i, answers[i]])]]
    return np.array(out_seq)

print("SQA dimensions:")
print(stories.shape)
print(queries.shape)
print(answers.shape)

word2int = pickle.load(open("../rc_data/word2int.pkl", "rb"))
int2word = {k: v for v, k in word2int.items()}
vocab_size = len(word2int)
print("vocab size = {0}".format(vocab_size))

n_train = stories.shape[0]
n_dev = dev_stories.shape[0]
entity_size = len(full_answers)
print("number of entities = {0}".format(entity_size))

entity2code = {w: i for w, i in zip(full_answers, range(entity_size))}
code2entity = {i: w for w, i in entity2code.items()}
answers_code = np.array([entity2code[x] for x in answers])
answers_code_dev = np.array([entity2code[x] for x in dev_answers])

batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]
batches += [(batches[-1][1], n_train)]
n_batches = len(batches)

dev_batches = zip(range(0, n_dev - batch_size, batch_size), range(batch_size, n_dev, batch_size))
dev_batches = [(start, end) for start, end in dev_batches]
dev_batches += [(dev_batches[-1][1], n_dev)]
dev_n_batches = len(dev_batches)

print("starting and final batches:")
print(batches[0])
print(batches[-1])

print("starting and final dev batches:")
print(dev_batches[0])
print(dev_batches[-1])

net = AttentiveReader(vocab_size, stories.shape[1], queries.shape[1], entity_size,
                      use_glove=True,
                      optimiser='adam',
                      learning_rate=0.001,
                      dropout=0.2,
                      init_range=0.01,
                      embedding_init_range=0.01,
                      max_grad_morm=10,
                      gru_init_std=0.1,
                      momentum=0.9,
                      hidden_size=128,
                      seed=seed
                      )
if use_cuda:
    net.cuda()
print("network initialised!")

if resume:
    print("loading saved states...")
    # resume test
    saved_state = torch.load("./checkpoint.packed.pth.tar")
    net.load_state_dict(saved_state['state_dict'])

print("testing multi-step training!")
ts = torch.from_numpy(stories).pin_memory()
tq = torch.from_numpy(queries).pin_memory()
ta = torch.from_numpy(answers_code).pin_memory()

dso = torch.from_numpy(dev_stories).pin_memory()
dqo = torch.from_numpy(dev_queries).pin_memory()
dao = torch.from_numpy(answers_code_dev).pin_memory()

if pack:
    ts_len = np.sum((stories > 0), 1)
    tq_len = np.sum((queries > 0), 1)
    ds_len = np.sum((dev_stories > 0), 1)
    dq_len = np.sum((dev_queries > 0), 1)
else:
    ds_len = None
    dq_len = None

net.train()
best_loss = np.inf
for epoch in range(n_epochs):
    print("epoch no.{0}".format(epoch + 1))

    np.random.shuffle(batches)
    for i in range(n_batches):
        start, end = batches[i]

        if use_cuda:
            bs = Variable(ts[start:end, :].cuda(async=True), requires_grad=False)
            bq = Variable(tq[start:end, :].cuda(async=True), requires_grad=False)
            ba = Variable(ta[start:end].cuda(async=True), requires_grad=False)
        else:
            bs = Variable(ts[start:end, :], requires_grad=False)
            bq = Variable(tq[start:end, :], requires_grad=False)
            ba = Variable(ta[start:end], requires_grad=False)
        if pack:
            bs_len = ts_len[start:end]
            bq_len = tq_len[start:end]
        else:
            bs_len = None
            bq_len = None

        loss, out = net.train_on_batch(bs, bq, bs_len, bq_len, ba, pack)
        if i % 100 == 1:
            net.eval()
            # print("iteration {0}/{1}\ntraining loss = {2:.10}".format(i, n_batches, loss.cpu().numpy()[0]))
            # _, out_seq = torch.max(out, 1)
            # out_seq = out_seq.data.cpu().numpy()
            out_seq = predict_in_domain(bs, out)
            acc = accuracy_score(out_seq, ba.data.cpu().numpy())
            print("iteration {0}/{1}\ntraining loss =   {2:.10}, training accuracy =   {3:.5}".
                  format(i, n_batches, loss.cpu().numpy()[0], acc))
            loss_score, acc_score = validate(net, dso, dqo, dao, ds_len, dq_len, dev_batches, cuda=use_cuda, pack=pack)
            print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                  format(loss_score.data.cpu().numpy()[0], acc_score))
            # print("validation accuracy = {0:.5}".format(acc_score))
            net.train()
        if i % 1000 == 1 or i == n_batches - 1:
            net.eval()
            val_loss, acc = validate(net, dso, dqo, dao, ds_len, dq_len, dev_batches, cuda=use_cuda, pack=pack)
            net.train()
            checkpoint_val_loss = val_loss.data.cpu().numpy()[0]
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
