import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from AttentiveReaderPlus import AttentiveReader
from AttentiveReaderBilingual import AttentiveReaderBilingual
from sklearn.metrics import accuracy_score
from data_utils import QADataset, get_embeddings, save_checkpoint, sort_batch, BiQADataset
import spacy
import argparse
import json
import os.path as path


def validate(network, dev_data_loader):
    n_dev_batches = len(dev_data_loader)
    total_val_loss = 0
    outputs = []
    ys = []
    for batch in dev_data_loader:
        s, q, sl, ql, sv, qv, t = sort_batch(batch, pack=params['pack'], sort_ind=3)
        s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        sl = Variable(sl, requires_grad=False)
        ql = Variable(ql, requires_grad=False)
        x = (s, q, sl, ql)
        y = Variable(t.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        out_logits = network.forward(x)

        out_seq = predict_in_domain(sv, out_logits)
        val_loss = nn.CrossEntropyLoss()(out_logits, y)
        total_val_loss += val_loss.data.cpu().numpy()[0]  # crucial!
        outputs += [out_seq]
        ys += [t.numpy()]
    loss_score = total_val_loss / n_dev_batches
    combined_outputs = np.concatenate(outputs, 0)
    ys = np.concatenate(ys, 0)
    acc_score = accuracy_score(combined_outputs, ys)
    return loss_score, acc_score


def validate_bireader(network, dev_data_loader, params):
    n_dev_batches = len(dev_data_loader)
    total_val_loss = 0
    total_ans_loss = 0
    total_dis_loss = 0
    outputs = []
    ys = []
    for batch in dev_data_loader:
        ln, s, q, sl, ql, sv, qv, t = sort_batch(batch, pack=params['pack'], sort_ind=4)
        ln = Variable(ln.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        sl = Variable(sl, requires_grad=False)
        ql = Variable(ql, requires_grad=False)
        x = (ln, s, q, sl, ql)
        y = Variable(t.type(torch.LongTensor).cuda(async=True), requires_grad=False)
        out_logits, discriminator_out = network.forward(x)
        out_seq = predict_in_domain(sv, out_logits)

        answerer_loss = nn.CrossEntropyLoss()(out_logits, y).data.cpu().numpy()[0]  # crucial!
        if params["adversarial"]:
            discriminator_loss = nn.BCEWithLogitsLoss()(discriminator_out, ln.float().unsqueeze(1))\
                .data.cpu().numpy()[0]  # crucial!
            val_loss = answerer_loss - params["discriminator_weight"] * discriminator_loss
            total_dis_loss += discriminator_loss
        else:
            val_loss = answerer_loss
        total_val_loss += val_loss
        total_ans_loss += answerer_loss
        outputs += [out_seq]
        ys += [t.numpy()]
    loss_score = total_val_loss / n_dev_batches
    ans_loss_score = total_ans_loss / n_dev_batches
    dis_loss_score = total_dis_loss / n_dev_batches
    combined_outputs = np.concatenate(outputs, 0)
    ys = np.concatenate(ys, 0)
    acc_score = accuracy_score(combined_outputs, ys)
    return loss_score, ans_loss_score, dis_loss_score, acc_score


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


def define_network(params, cross=None, requires_optim=True):
    if cross:
        lang = cross
    else:
        lang = params['lang']
    selected_embedding = params['selected_embedding']
    # load new vectors
    # with open("../rc_data/processed/glove.6B/glove.6B.100d.txt".format(lang), "r") as f:
    with open("../bilingual_vector/{1}/wiki.{0}.small.vec".format(lang,
                                                                  selected_embedding), "r") as f:
        emb_vectors, dic, rev_dic = get_embeddings(f, nr_unk=100, nr_var=600, meta=params['embedding_meta'])
    print("embedding {1}/wiki.{0}.small.vec loaded!".format(lang, selected_embedding))

    net = AttentiveReader(params['var_size'], 2000, 50, emb_vectors,
                          dropout=params['dropout'],
                          hidden_size=params['hidden_size'],
                          pack=params['pack'],
                          emb_trainable=params['emb_trainable'],
                          story_rec_layers=params['story_rec_layers'])

    if requires_optim:
        if params['optimiser'] == 'sgd/momentum':
            net.optimiser = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=params['learning_rate'],
                                      momentum=params['momentum'])
        elif params['optimiser'] == 'sgd':
            net.optimiser = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=params['learning_rate'],
                                      momentum=0)
        elif params['optimiser'] == 'adam':
            net.optimiser = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=params['learning_rate'])
        elif params['optimiser'] == 'adadelta':
            net.optimiser = optim.Adadelta(filter(lambda p: p.requires_grad, net.parameters()),
                                           lr=params['learning_rate'])
        else:
            raise NotImplementedError("Invalid optimiser!")

    net.cuda()
    print("network initialised!")
    return net, dic, rev_dic


def define_bilingual_network(params, l2='es', requires_optim=True):
    l1 = params['lang']
    selected_embedding = params['selected_embedding']
    with open("../bilingual_vector/{1}/wiki.{0}.small.vec".format(l1,
                                                                  selected_embedding), "r") as f:
        emb_vectors_1, dic_1, rev_dic_1 = get_embeddings(f, nr_unk=100, nr_var=600, meta=params['embedding_meta'])
    print("embedding {1}/wiki.{0}.small.vec loaded!".format(l1, selected_embedding))
    with open("../bilingual_vector/{1}/wiki.{0}.small.vec".format(l2,
                                                                  selected_embedding), "r") as f:
        emb_vectors_2, dic_2, rev_dic_2 = get_embeddings(f, nr_unk=100, nr_var=600, meta=params['embedding_meta'])
    print("embedding {1}/wiki.{0}.small.vec loaded!".format(l2, selected_embedding))

    net = AttentiveReaderBilingual(params['var_size'], 2000, 50, (emb_vectors_1, emb_vectors_2),
                                   dropout=params['dropout'],
                                   hidden_size=params['hidden_size'],
                                   pack=params['pack'],
                                   emb_trainable=params['emb_trainable'],
                                   story_rec_layers=params['story_rec_layers'],
                                   adversarial=params['adversarial'],
                                   discriminator_hidden_size=params["discriminator_hidden_size"],
                                   discriminator_weight=params["discriminator_weight"],
                                   )

    if requires_optim:
        if params['optimiser'] == 'sgd/momentum':
            net.optimiser = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=params['learning_rate'],
                                      momentum=params['momentum'])
        elif params['optimiser'] == 'sgd':
            net.optimiser = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=params['learning_rate'],
                                      momentum=0)
        elif params['optimiser'] == 'adam':
            net.optimiser = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=params['learning_rate'])
        elif params['optimiser'] == 'adadelta':
            net.optimiser = optim.Adadelta(filter(lambda p: p.requires_grad, net.parameters()),
                                           lr=params['learning_rate'])
        else:
            raise NotImplementedError("Invalid optimiser!")

    net.cuda()
    print("network initialised!")
    return net, dic_1, dic_2, rev_dic_1, rev_dic_2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='.', help='Parameters file directory')
    parser.add_argument('-v', '--validate', action='store_true')
    parser.add_argument('-c', '--cross', type=str, help="target language for cross-lingual evaluation")
    parser.add_argument('-t', '--crosstrain', type=str, help="target language for cross-lingual training")
    parser.add_argument('-b', '--bilingual', action='store_true')
    arg = parser.parse_args()
    params_file = path.join(arg.params, "params.json")
    params = json.load(open(params_file, "rb"))
    print(params)
    checkpoint_file = path.join(arg.params, "checkpoint.{0}.pth.tar".format(params['lang']))
    best_file = path.join(arg.params, "model_best.{0}.pth.tar".format(params['lang']))

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    if arg.bilingual:
        print("starting bilingual training...")
        print("building network...")
        net, _, _, rev_dic_1, rev_dic_2 = define_bilingual_network(params, params['lang2'], requires_optim=True)
        print("network defined!")
        nlp_1 = spacy.load(params['lang'], vectors=False)
        nlp_2 = spacy.load(params['lang2'], vectors=False)
        print("loading data...")
        if arg.validate:
            print("now validating...")
            dev_1 = pd.read_pickle("../input_data/dev_{0}.pkl".format(params['lang']))
            dev_2 = pd.read_pickle("../input_data/dev_{0}.pkl".format(params['lang2']))
            dev_loader = tud.DataLoader(BiQADataset(dev_1, dev_2, nlp_1, nlp_2,
                                                    rev_dic_1, rev_dic_2, relabel=params['relabel']),
                                        batch_size=params['batch_size'],
                                        pin_memory=True, num_workers=3)
            test_1 = pd.read_pickle("../input_data/test_{0}.pkl".format(params['lang']))
            test_2 = pd.read_pickle("../input_data/test_{0}.pkl".format(params['lang2']))
            test_loader = tud.DataLoader(BiQADataset(test_1, test_2, nlp_1, nlp_2,
                                                     rev_dic_1, rev_dic_2, relabel=params['relabel']),
                                         batch_size=params['batch_size'],
                                         pin_memory=True, num_workers=3)
            checkpoint_file = path.join(arg.params, "checkpoint.{0}.{1}.pth.tar".format(params['lang'],
                                                                                        params['lang2']))
            best_file = path.join(arg.params, "model_best.{0}.{1}.pth.tar".format(params['lang'], params['lang2']))
            print("loading states...")
            best_state = torch.load(best_file)
            net.load_state_dict(best_state['state_dict'])
            del best_state
            print("evaluating validation performance...")
            net.eval()
            loss_score, ans_loss_score, dis_loss_score, acc_score = validate_bireader(net, dev_loader, params)
            print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                  format(loss_score, acc_score))
            print("answerer loss = {0:.10}, discrim. loss = {1:.10}".
                  format(ans_loss_score, dis_loss_score))
            loss_score, ans_loss_score, dis_loss_score, acc_score = validate_bireader(net, test_loader, params)
            print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                  format(loss_score, acc_score))
            print("answerer loss = {0:.10}, discrim. loss = {1:.10}".
                  format(ans_loss_score, dis_loss_score))

        else:
            print("now training...")
            train_1 = pd.read_pickle("../input_data/train_{0}.pkl".format(params['lang']))
            train_2 = pd.read_pickle("../input_data/train_{0}.pkl".format(params['lang2']))
            train_loader = tud.DataLoader(BiQADataset(train_1, train_2, nlp_1, nlp_2,
                                                      rev_dic_1, rev_dic_2, relabel=params['relabel'],
                                                      l2_supersample=params['l2_supersample']),
                                          batch_size=params['batch_size'],
                                          pin_memory=True, num_workers=3,
                                          shuffle=True)

            dev_1 = pd.read_pickle("../input_data/dev_{0}.pkl".format(params['lang']))
            dev_2 = pd.read_pickle("../input_data/dev_{0}.pkl".format(params['lang2']))
            dev_loader = tud.DataLoader(BiQADataset(dev_1, dev_2, nlp_1, nlp_2,
                                                    rev_dic_1, rev_dic_2, relabel=params['relabel'],
                                                    l2_supersample=0  # debug
                                                    ),
                                        batch_size=params['batch_size'],
                                        pin_memory=True, num_workers=3)
            # best_acc = 0
            best_loss = np.inf
            init_epoch = 0
            checkpoint_file = path.join(arg.params, "checkpoint.{0}.{1}.pth.tar".format(params['lang'], params['lang2']))
            best_file = path.join(arg.params, "model_best.{0}.{1}.pth.tar".format(params['lang'], params['lang2']))

            print("starting training!")
            for epoch in range(init_epoch, params['n_epochs']):
                print("epoch no.{0} start!".format(epoch + 1))

                i = 0
                cum_loss = 0
                cum_ans_loss = 0
                cum_dis_loss = 0
                for i, batch in enumerate(train_loader):

                    ln, s, q, sl, ql, sv, qv, y = sort_batch(batch, pack=params['pack'], sort_ind=4)
                    ln = Variable(ln.type(torch.LongTensor).cuda(async=True), requires_grad=False)
                    s = Variable(s.type(torch.LongTensor).cuda(async=True), requires_grad=False)
                    q = Variable(q.type(torch.LongTensor).cuda(async=True), requires_grad=False)
                    sl = Variable(sl, requires_grad=False)
                    ql = Variable(ql, requires_grad=False)
                    y = Variable(y.type(torch.LongTensor).cuda(async=True), requires_grad=False)
                    xy = (ln, s, q, sl, ql, y)

                    net.train()
                    loss, _, ans_loss, dis_loss = net.train_on_batch(xy)
                    cum_loss += loss
                    cum_ans_loss += ans_loss
                    cum_dis_loss += dis_loss

                    if i % 100 == 1:
                        net.eval()
                        if i == 1:
                            cum_loss *= 50
                            cum_ans_loss *= 50
                            cum_dis_loss *= 50
                        print("iteration {0}/{1}\ntraining loss =    {2:.10}\nanswerer loss =    {3:.10}\n"
                              "discriminator loss={4:.10}".
                              format(i, len(train_loader), cum_loss / 100., cum_ans_loss / 100., cum_dis_loss / 100.))
                        cum_loss = 0
                        cum_ans_loss = 0
                        cum_dis_loss = 0
                    if i % 1000 == 1 or i == len(train_loader) - 1:
                        net.eval()
                        loss_score, ans_loss_score, dis_loss_score, acc_score = \
                            validate_bireader(net, dev_loader, params)
                        print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                              format(loss_score, acc_score))
                        print("answerer loss = {0:.10}, discrim. loss = {1:.10}".
                              format(ans_loss_score, dis_loss_score))
                        is_best = loss_score < best_loss
                        if is_best:
                            best_loss = loss_score
                        save_checkpoint(
                            {
                                'epoch': epoch,
                                'batch': i,
                                'state_dict': net.state_dict(),
                                'epoch_val_loss': loss_score,
                                'acc': acc_score
                            },
                            is_best,
                            filename=checkpoint_file,
                            best_name=best_file)

    elif arg.validate and arg.cross is None:
        print("validating...")
        print("building network...")
        nlp = spacy.load(params['lang'], vectors=False)
        net, dic, rev_dic = define_network(params, requires_optim=False)
        print("loading saved states...")
        best_state = torch.load(best_file)
        net.load_state_dict(best_state['state_dict'])
        del best_state
        print("loading validation data...")
        dev = pd.read_pickle("../input_data/dev_{0}.pkl".format(params['lang']))
        dev_loader = tud.DataLoader(QADataset(dev, nlp, rev_dic, relabel=params['relabel']),
                                    batch_size=params['batch_size'],
                                    pin_memory=True, num_workers=3)
        test = pd.read_pickle("../input_data/test_{0}.pkl".format(params['lang']))
        test_loader = tud.DataLoader(QADataset(test, nlp, rev_dic, relabel=params['relabel']),
                                     batch_size=params['batch_size'],
                                     pin_memory=True, num_workers=3, shuffle=True)
        print("evaluating validation performance...")
        net.eval()
        loss_score, acc_score = validate(net, dev_loader)
        print("validation loss = {0:.10}, validation accuracy = {1:.5}".
              format(loss_score, acc_score))
        loss_score, acc_score = validate(net, test_loader)
        print("test loss = {0:.10}, test accuracy = {1:.5}".
              format(loss_score, acc_score))

    elif arg.cross is not None:
        print("validating cross-lingual performance...")
        print("building network...")
        nlp = spacy.load(arg.cross, vectors=False)
        if arg.validate:
            # point to new checkpoint files
            checkpoint_file = path.join(arg.params, "cross_checkpoint.{0}.pth.tar".format(arg.cross))
            best_file = path.join(arg.params, "cross_model_best.{0}.pth.tar".format(arg.cross))
            net, dic, rev_dic = define_network(params, cross=arg.cross, requires_optim=False)
            print("loading saved states...")
            best_state = torch.load(best_file)
            net.load_state_dict(best_state['state_dict'])
            del best_state
        else:
            # dummy network
            dnet, _, _ = define_network(params, requires_optim=False)
            best_state = torch.load(best_file)
            dnet.load_state_dict(best_state['state_dict'])
            # real network
            net, dic, rev_dic = define_network(params, cross=arg.cross, requires_optim=False)
            net.set_weights_except_embeddings(dnet.get_weights())
            del dnet, best_state

        dev = pd.read_pickle("../input_data/dev_{0}.pkl".format(arg.cross))
        dev_loader = tud.DataLoader(QADataset(dev, nlp, rev_dic, relabel=params['relabel']),
                                    batch_size=params['batch_size'],
                                    pin_memory=True, num_workers=3)
        test = pd.read_pickle("../input_data/test_{0}.pkl".format(arg.cross))
        test_loader = tud.DataLoader(QADataset(test, nlp, rev_dic, relabel=params['relabel']),
                                     batch_size=params['batch_size'],
                                     pin_memory=True, num_workers=3)
        print("evaluating validation performance...")
        net.eval()
        loss_score, acc_score = validate(net, dev_loader)
        print("validation loss = {0:.10}, validation accuracy = {1:.5}".
              format(loss_score, acc_score))
        loss_score, acc_score = validate(net, test_loader)
        print("test loss = {0:.10}, test accuracy = {1:.5}".
              format(loss_score, acc_score))

    elif arg.crosstrain is not None:
        print("starting cross-lingual training...")
        print("building network...")
        nlp = spacy.load(arg.crosstrain, vectors=False)
        # dummy network
        dnet, _, _ = define_network(params, requires_optim=False)
        best_state = torch.load(best_file)
        dnet.load_state_dict(best_state['state_dict'])
        # real network
        net, dic, rev_dic = define_network(params, cross=arg.crosstrain, requires_optim=True)
        net.set_weights_except_embeddings(dnet.get_weights())
        del dnet, best_state

        dev = pd.read_pickle("../input_data/dev_{0}.pkl".format(arg.crosstrain))
        dev_loader = tud.DataLoader(QADataset(dev, nlp, rev_dic, relabel=params['relabel']),
                                    batch_size=params['batch_size'],
                                    pin_memory=True, num_workers=3)
        train = pd.read_pickle("../input_data/train_{0}.pkl".format(arg.crosstrain))
        train_loader = tud.DataLoader(QADataset(train, nlp, rev_dic, relabel=params['relabel']),
                                      batch_size=params['batch_size'],
                                      pin_memory=True, num_workers=3, shuffle=True)
        best_loss = np.inf
        # best_acc = 0
        init_epoch = 0
        # point to new checkpoint files
        checkpoint_file = path.join(arg.params, "cross_checkpoint.{0}.pth.tar".format(arg.crosstrain))
        best_file = path.join(arg.params, "cross_model_best.{0}.pth.tar".format(arg.crosstrain))
        # TODO: allow cross-training to be resumed
        # if params['resume']:
        #     print("loading saved states...")
        #     # resume test
        #     saved_state = torch.load(checkpoint_file)
        #     best_state = torch.load(best_file)
        #     net.load_state_dict(saved_state['state_dict'])
        #     init_epoch = saved_state['epoch']
        #     best_loss = best_state['epoch_val_loss']
        #     del best_state

        print("starting training!")
        # net.bypass_softmax = True
        for epoch in range(init_epoch, params['n_epochs']):
            print("epoch no.{0} start!".format(epoch + 1))

            # if epoch > 1:
            #     net.bypass_softmax = False

            i = 0
            cum_loss = 0
            for i, batch in enumerate(train_loader):

                s, q, sl, ql, sv, qv, y = sort_batch(batch, pack=params['pack'], sort_ind=3)
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
                    print("epoch no.{0}".format(epoch + 1))
                    print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                          format(val_loss, acc))
                    # checkpoint_val_loss = val_loss
                    is_best = val_loss < best_loss
                    # is_best = acc > best_acc
                    if is_best:
                        best_loss = val_loss
                        # best_acc = acc
                    # print("checkpoint loss = {0:.10}".format(checkpoint_val_loss))
                    save_checkpoint(
                        {
                            'epoch': epoch,
                            'batch': i,
                            'state_dict': net.state_dict(),
                            'epoch_val_loss': val_loss,
                            'acc': acc
                        },
                        is_best,
                        filename=checkpoint_file,
                        best_name=best_file)

    else:
        print("building network...")
        nlp = spacy.load(params['lang'], vectors=False)
        net, dic, rev_dic = define_network(params)
        train = pd.read_pickle("../input_data/train_{0}.pkl".format(params['lang']))
        dev = pd.read_pickle("../input_data/dev_{0}.pkl".format(params['lang']))
        train_loader = tud.DataLoader(QADataset(train, nlp, rev_dic, relabel=params['relabel']),
                                      batch_size=params['batch_size'], pin_memory=True,
                                      num_workers=3, shuffle=True)
        dev_loader = tud.DataLoader(QADataset(dev, nlp, rev_dic, relabel=params['relabel']),
                                    batch_size=params['batch_size'],
                                    pin_memory=True, num_workers=3)

        best_loss = np.inf
        # best_acc = 0
        init_epoch = 0
        if params['resume']:
            print("loading saved states...")
            # resume test
            saved_state = torch.load(checkpoint_file)
            best_state = torch.load(best_file)
            net.load_state_dict(saved_state['state_dict'])
            init_epoch = saved_state['epoch']
            best_loss = best_state['epoch_val_loss']
            # best_acc = best_state['acc']
            del best_state

        print("testing multi-step training!")
        # TODO: add to other parameter settings
        net.bypass_softmax = True
        for epoch in range(init_epoch, params['n_epochs']):
            print("epoch no.{0} start!".format(epoch + 1))
            # if epoch > 1:
            #     net.bypass_softmax = False
            i = 0
            cum_loss = 0
            for i, batch in enumerate(train_loader):

                s, q, sl, ql, sv, qv, y = sort_batch(batch, pack=params['pack'], sort_ind=3)
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
                    print("epoch no.{0}".format(epoch + 1))
                    print("validation loss = {0:.10}, validation accuracy = {1:.5}".
                          format(val_loss, acc))
                    # checkpoint_val_loss = val_loss
                    is_best = val_loss < best_loss
                    # is_best = acc > best_acc
                    if is_best:
                        best_loss = val_loss
                        # best_acc = acc
                    # print("checkpoint loss = {0:.10}".format(checkpoint_val_loss))
                    save_checkpoint(
                        {
                            'epoch': epoch,
                            'batch': i,
                            'state_dict': net.state_dict(),
                            'epoch_val_loss': val_loss,
                            'acc': acc
                        },
                        is_best,
                        filename=checkpoint_file,
                        best_name=best_file)
