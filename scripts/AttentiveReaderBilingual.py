import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence


class AttentiveReaderBilingual(nn.Module):
    """Attentive Reader"""
    def __init__(self, var_size, story_max_len, question_max_len, emb_vectors,
                 hidden_size=128,
                 dropout=0.2,
                 max_grad_norm=10,
                 nr_unk=100,
                 pack=True,
                 emb_trainable=False,
                 gru_init_std=0.1,
                 init_range=0.01,
                 story_rec_layers=1,
                 adversarial=True,
                 discriminator_hidden_size=128,
                 discriminator_weight=0.5,
                 opt=None,
                 adv_opt=None):
        super(AttentiveReaderBilingual, self).__init__()

        # setting attributes
        self._var_size = var_size
        self._gru_init_std = gru_init_std
        self._init_range = init_range
        self._story_size = story_max_len
        self._question_size = question_max_len
        self._hidden_size = hidden_size
        self._dropout_rate = dropout
        self._l1_emb_vector, self._l2_emb_vector = emb_vectors
        self._max_grad_norm = max_grad_norm
        self._pack = pack
        self._nr_unk = nr_unk
        self._emb_trainable = emb_trainable
        self._story_rec_layers = story_rec_layers
        self._adversarial = adversarial
        self._discriminator_hidden_size = discriminator_hidden_size
        self._discriminator_weight = discriminator_weight
        self.optimiser = opt
        self.adv_optimiser = adv_opt

        # create layers

        # variable embeddings
        assert(self._l1_emb_vector.shape[1] == self._l2_emb_vector.shape[1])
        self._l1_embedding_layer = nn.Embedding(self._l1_emb_vector.shape[0], self._l1_emb_vector.shape[1], 0)
        self._l2_embedding_layer = nn.Embedding(self._l2_emb_vector.shape[0], self._l2_emb_vector.shape[1], 0)

        self._dropout = nn.Dropout(dropout)
        self._recurrent_layer = nn.GRU(self._l1_emb_vector.shape[1], hidden_size, story_rec_layers,
                                       batch_first=True,
                                       bidirectional=True)
        self._question_recurrent_layer = nn.GRU(self._l1_emb_vector.shape[1], hidden_size, 1,
                                                batch_first=True,
                                                bidirectional=True)

        self._output_layer = nn.Linear(hidden_size * 2, var_size)
        self._mix_matrix = nn.Parameter(torch.zeros((hidden_size * 2, hidden_size * 2)))

        if adversarial:
            self._discriminator = nn.Sequential(nn.Linear(hidden_size * 2, discriminator_hidden_size),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Linear(discriminator_hidden_size, 1)
                                                )

        self.init_weights()

    def init_weights(self):
        init_range = self._init_range
        init_std = self._gru_init_std

        self._l1_embedding_layer.weight.data.copy_(torch.from_numpy(self._l1_emb_vector))
        self._l2_embedding_layer.weight.data.copy_(torch.from_numpy(self._l2_emb_vector))

        unk_n_var_1 = self._l1_embedding_layer.weight.data[1: 2 + self._nr_unk + self._var_size, :]
        init.normal(unk_n_var_1, 0, 1)
        unk_n_var_1 /= torch.norm(unk_n_var_1, p=2, dim=1).unsqueeze(1)  # normalise randomly initialised embeddings
        self._l1_embedding_layer.weight.data[0, :] = 0
        if not self._emb_trainable:
            self._l1_embedding_layer.weight.requires_gard = False  # size = entities + ph + non-ent-marker

        unk_2 = self._l2_embedding_layer.weight.data[2: 2 + self._nr_unk, :]
        init.normal(unk_2, 0, 1)
        unk_2 /= torch.norm(unk_2, p=2, dim=1).unsqueeze(1)  # normalise randomly initialised embeddings
        # ^^^ init unk * 100 embeddings
        self._l2_embedding_layer.weight.data[1, :] = self._l1_embedding_layer.weight.data[1, :]
        # ^^^ share @placeholder embedding
        self._l2_embedding_layer.weight.data[2 + self._nr_unk: 2 + self._nr_unk + self._var_size, :] = \
            self._l1_embedding_layer.weight.data[2 + self._nr_unk: 2 + self._nr_unk + self._var_size, :]
        # ^^^ share @entityX embeddings
        self._l2_embedding_layer.weight.data[0, :] = 0
        if not self._emb_trainable:
            self._l2_embedding_layer.weight.requires_gard = False  # size = entities + ph + non-ent-marker
        # DONE: initialise non-zero locations
        # TODO: randomise in forward step?

        gain = init.calculate_gain('tanh')
        for p in self._recurrent_layer.parameters():
            if p.dim() == 1:
                p.data.normal_(0, init_std)
            else:
                init.orthogonal(p.data, gain)
        for p in self._question_recurrent_layer.parameters():
            if p.dim() == 1:
                p.data.normal_(0, init_std)
            else:
                init.orthogonal(p.data, gain)

        # self._embedding_projection_layer.weight.data.uniform_(-init_range, init_range)
        self._output_layer.weight.data.uniform_(-init_range, init_range)
        self._output_layer.bias.data.fill_(0)

        self._mix_matrix.data.uniform_(-init_range, init_range)

    def forward(self, batch):
        lang, story, question, story_len, question_len = batch
        batch_size = story.size()[0]

        lang = lang.unsqueeze(1)
        s_emb = self._dropout(self._l1_embedding_layer(story * (1 + (-1) * lang))
                              + self._l2_embedding_layer(story * lang))

        q_emb = self._dropout(self._l1_embedding_layer(question * (1 + (-1) * lang))
                              + self._l2_embedding_layer(question * lang))

        if self._pack:
            # TODO: still have to sort questions by length after mixing two languages
            q_emb = pack_padded_sequence(q_emb, question_len.data.numpy(), batch_first=True)
            # ^^^ use this line of only batching questions

        y_out, _ = self._recurrent_layer(s_emb)  # batch * story_size * 2hidden_size
        _, q_hn = self._question_recurrent_layer(q_emb)  # _, batch * 2hidden_size

        q_hn = q_hn.permute(1, 0, 2).contiguous().view((batch_size, self._hidden_size * 2, 1))

        ms = y_out.bmm(self._mix_matrix.unsqueeze(0).expand(batch_size,
                                                            self._hidden_size * 2,
                                                            self._hidden_size * 2))  # batched matrix
        ms = ms.bmm(q_hn)  # batched [col, col, col, ...] -> batched [scalar, scalar, scalar, ...]

        ss = F.softmax(ms)

        r = torch.sum(y_out * ss.expand_as(y_out), dim=1, keepdim=True)  # batch * 2hidden_size
        out = self._output_layer(r.squeeze())

        # TODO: try different locations to insert adversarial network
        if self._adversarial:
            discriminator_out = self._discriminator(r)
            return out, discriminator_out
        else:
            return out, None

    def train_on_batch(self, batch):
        """call net.train() before calling this method!"""
        out, discriminator_out = self.forward(batch[:-1])
        answers = batch[-1]
        # TODO: use adversarial loss if enabled
        if self._adversarial:
            lang = batch[0]
            # loss = nn.CrossEntropyLoss()(out, answers)\
            #     - self._discriminator_weight * nn.BCEWithLogitsLoss()(discriminator_out, lang.unsqueeze(1))
            answerer_loss = nn.CrossEntropyLoss()(out, answers)
            discriminator_loss = nn.BCEWithLogitsLoss()(discriminator_out, lang.float().unsqueeze(1))
            loss = answerer_loss - self._discriminator_weight * discriminator_loss
        else:
            loss = nn.CrossEntropyLoss()(out, answers)
        self.zero_grad()
        loss.backward()
        self._reset_nil_gradients()
        torch.nn.utils.clip_grad_norm(self.parameters(), self._max_grad_norm)
        if self._adversarial:
            # negate the gradient of the discriminator so we minimise the discriminator loss wrt discriminator weights
            for p in self._discriminator.parameters():
                p._grad = -p._grad
        self.optimiser.step()
        if self._adversarial:
            return loss.data.cpu().numpy()[0], \
                   answerer_loss.data.cpu().numpy()[0], discriminator_loss.data.cpu().numpy()[0]
        else:
            return loss.data.cpu().numpy()[0], loss.data.cpu().numpy()[0], 0

    # def predict(self, batch):
    #     """call net.eval() before calling this method!"""
    #     out, _ = self.forward(batch)
    #     return F.softmax(out)

    def _reset_nil_gradients(self):
        if self._emb_trainable:
            self._embedding_layer.weight.data[0, :] = 0

    def get_weights(self):
        return (self._l1_embedding_layer.weight.data, self._l2_embedding_layer.weight.data,
                self._recurrent_layer.parameters(), self._question_recurrent_layer.parameters(),
                self._output_layer.parameters(), self._mix_matrix.data)

    def set_weights_except_embeddings(self, weights):
        ew1, ew2, rws, qws, ows, mw = weights
        for w, t in zip(self._recurrent_layer.parameters(), rws):
            w.data.copy_(t.data)
        for w, t in zip(self._question_recurrent_layer.parameters(), qws):
            w.data.copy_(t.data)
        for w, t in zip(self._output_layer.parameters(), ows):
            w.data.copy_(t.data)
        self._mix_matrix.data.copy_(mw)
        self._l1_embedding_layer.weight.data[1: 2 + self._nr_unk + self._var_size, :] = \
            ew1[1: 2 + self._nr_unk + self._var_size, :]
        self._l2_embedding_layer.weight.data[1: 2 + self._nr_unk + self._var_size, :] = \
            ew2[1: 2 + self._nr_unk + self._var_size, :]

    # @staticmethod
    # def softmax(inputs, dim=1):
    #     input_size = inputs.size()
    #
    #     trans_input = inputs.transpose(dim, len(input_size) - 1)
    #     trans_size = trans_input.size()
    #
    #     input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    #
    #     soft_max_2d = F.softmax(input_2d)
    #
    #     soft_max_nd = soft_max_2d.view(*trans_size)
    #     return soft_max_nd.transpose(dim, len(input_size) - 1)
