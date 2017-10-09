import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentiveReader(nn.Module):
    """Attentive Reader"""
    def __init__(self, var_size, story_max_len, question_max_len, emb_vectors,
                 hidden_size=128,
                 dropout=0.2,
                 max_grad_norm=10,
                 nr_unk=100,
                 pack=True,
                 emb_trainable=True,
                 gru_init_std=0.1,
                 init_range=0.01,
                 story_rec_layers=1,
                 opt=None):
        super(AttentiveReader, self).__init__()

        # setting attributes
        self._var_size = var_size
        self._gru_init_std = gru_init_std
        self._init_range = init_range
        self._story_size = story_max_len
        self._question_size = question_max_len
        self._hidden_size = hidden_size
        self._dropout_rate = dropout
        self._emb_vector = emb_vectors
        self._max_grad_norm = max_grad_norm
        self._pack = pack
        self._nr_unk = nr_unk
        self._emb_trainable = emb_trainable
        self._story_rec_layers = story_rec_layers
        self.optimiser = opt

        # create layers

        # variable embeddings
        self._embedding_layer = nn.Embedding(emb_vectors.shape[0], emb_vectors.shape[1], 0)


        if not emb_trainable:
            self._embedding_layer.weight.requires_gard = False
        # ^^^ size = entities + ph + non-ent-marker
        # DONE: initialise non-zero locations
        # TODO: randomise in forward step?

        self._dropout = nn.Dropout(dropout)
        self._recurrent_layer = nn.GRU(emb_vectors.shape[1], hidden_size, story_rec_layers,
                                       batch_first=True,
                                       bidirectional=True)
        self._question_recurrent_layer = nn.GRU(emb_vectors.shape[1], hidden_size, 1,
                                                batch_first=True,
                                                bidirectional=True)
        self._output_layer = nn.Linear(hidden_size * 2, var_size)
        self._mix_matrix = nn.Parameter(torch.zeros((hidden_size * 2, hidden_size * 2)))

        self.init_weights()

    def init_weights(self):
        init_range = self._init_range
        init_std = self._gru_init_std

        self._embedding_layer.weight.data.copy_(torch.from_numpy(self._emb_vector))
        # self._embedding_layer.weight.data[1: 2 + self._nr_unk + self._var_size, :].normal_(0, 1)
        unk_n_var = self._embedding_layer.weight.data[1: 2 + self._nr_unk + self._var_size, :]
        init.normal(unk_n_var, 0, 1)
        unk_n_var /= torch.norm(unk_n_var, p=2, dim=1).unsqueeze(1)  # normalise randomly initialised embeddings
        # ^^^ init unk * 100 embeddings
        self._embedding_layer.weight.data[0, :] = 0

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

        self._output_layer.weight.data.uniform_(-init_range, init_range)
        self._output_layer.bias.data.fill_(0)

        self._mix_matrix.data.uniform_(-init_range, init_range)

    def forward(self, batch):
        story, question, story_len, question_len = batch
        batch_size = story.size()[0]

        s_emb = self._dropout(self._embedding_layer(story))

        q_emb = self._dropout(self._embedding_layer(question))

        if self._pack:
            # pack sequences
            s_emb = pack_padded_sequence(s_emb, story_len.data.numpy(), batch_first=True)

            queries_len_ordered, queries_len_order = torch.sort(question_len, descending=True)
            # ^^^ # get question lengths in sorted order, get order of question length
            _, queries_inv_order = torch.sort(queries_len_order)
            queries_len_order = queries_len_order.cuda()
            queries_inv_order = queries_inv_order.cuda()
            q_emb = q_emb.index_select(0, queries_len_order)  # sort embeddings in question length order
            q_emb = pack_padded_sequence(q_emb, queries_len_ordered.data.numpy(), batch_first=True)

        y_out, _ = self._recurrent_layer(s_emb)  # batch * story_size * 2hidden_size
        _, q_hn = self._question_recurrent_layer(q_emb)  # _, batch * 2hidden_size

        if self._pack:
            y_out, _ = pad_packed_sequence(y_out, batch_first=True)

        # left = y_out.contiguous()  #  seems to be no longer needed

        q_hn = q_hn.permute(1, 0, 2).contiguous().view((batch_size, self._hidden_size * 2, 1))

        if self._pack:
            q_hn = q_hn.index_select(0, queries_inv_order)  # batched rows -> reorder batches

        ms = y_out.bmm(self._mix_matrix.unsqueeze(0).expand(batch_size,
                                                            self._hidden_size * 2,
                                                            self._hidden_size * 2))  # batched matrix
        ms = ms.bmm(q_hn)  # batched [col, col, col, ...] -> batched [scalar, scalar, scalar, ...]
        ss = F.softmax(ms)

        r = torch.sum(y_out * ss.expand_as(y_out), dim=1, keepdim=True)  # batch * 2hidden_size
        out = self._output_layer(r.squeeze())
        return out

    def train_on_batch(self, batch):
        """call net.train() before calling this method!"""
        out = self.forward(batch[:-1])
        answers = batch[-1]
        loss = nn.CrossEntropyLoss()(out, answers)
        self.zero_grad()
        loss.backward()
        self._reset_nil_gradients()
        torch.nn.utils.clip_grad_norm(self.parameters(), self._max_grad_norm)
        self.optimiser.step()
        return loss.data, F.softmax(out)

    def predict(self, batch):
        """call net.eval() before calling this method!"""
        out = self.forward(batch)
        return F.softmax(out)

    def _reset_nil_gradients(self):
        if self._emb_trainable:
            self._embedding_layer.weight.data[0, :] = 0

    @staticmethod
    def softmax(inputs, axis=1):
        input_size = inputs.size()

        trans_input = inputs.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()

        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)

        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)