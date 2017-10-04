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
                 opt=None):
        super(AttentiveReader, self).__init__()

        # setting attributes
        self._var_size = var_size
        self._story_size = story_max_len
        self._question_size = question_max_len
        self._hidden_size = hidden_size
        self._dropout_rate = dropout
        self._emb_vector = emb_vectors
        self._max_grad_norm = 10
        self.optimiser = opt

        # create layers

        # variable embeddings
        self._embedding_layer = nn.Embedding(emb_vectors.shape[0], emb_vectors.shape[1], 0)
        self._embedding_layer.weight.data.copy_(torch.from_numpy(emb_vectors))
        # self._embedding_layer.weight.data[:emb_vectors.shape[0], :] = torch.from_numpy(emb_vectors)
        self._embedding_layer.weight.requires_gard = False

        self._var_embedding_layer = nn.Embedding(var_size + 2, emb_vectors.shape[1], 0)
        # ^^^ size = entities + ph + non-ent-marker
        # DONE: initialise non-zero locations
        # TODO: randomise in forward step?

        self._dropout = nn.Dropout(dropout)
        self._recurrent_layer = nn.GRU(emb_vectors.shape[1], hidden_size, 1,
                                       batch_first=True,
                                       bidirectional=True)
        self._question_recurrent_layer = nn.GRU(emb_vectors.shape[1], hidden_size, 1,
                                                batch_first=True,
                                                bidirectional=True)
        self._output_layer = nn.Linear(hidden_size * 2, var_size)
        self._mix_matrix = nn.Parameter(torch.zeros((hidden_size * 2, hidden_size * 2)))

        # fix embedding of non-entities and paddings to zero
        self._var_embedding_layer.weight.data[0, :] = 0
        # self._embedding_layer.weight.data[0, :] = 0

    def forward(self, batch):
        story, question, story_len, question_len, story_vocab_mask, question_vocab_mask, \
            story_vars, question_vars = batch
        batch_size = story.size()[0]

        s_emb = self._embedding_layer(story)
        s_var_emb = self._var_embedding_layer(story_vars)
        s_emb += s_var_emb

        q_emb = self._embedding_layer(question)
        q_var_emb = self._var_embedding_layer(question_vars)
        q_emb += q_var_emb

        # pack sequences
        s_emb = pack_padded_sequence(s_emb, story_len.data.numpy(), batch_first=True)
        # TODO: check if all word indices are in range

        _, queries_len_order = torch.sort(question_len, descending=True)  # get order of question length
        _, queries_inv_order = torch.sort(queries_len_order)
        q_emb = q_emb.index_select(0, queries_len_order.cuda())  # sort embeddings in question length order
        queries_len_ordered, _ = torch.sort(question_len, descending=True)  # get question lengths in sorted order
        q_emb = pack_padded_sequence(q_emb, queries_len_ordered.data.cpu().numpy(), batch_first=True)

        y_out, _ = self._recurrent_layer(s_emb)  # batch * story_size * 2hidden_size
        _, q_hn = self._question_recurrent_layer(q_emb)  # _, batch * 2hidden_size

        y_out, _ = pad_packed_sequence(y_out, batch_first=True)

        left = y_out.contiguous()
        q_hn = q_hn.permute(1, 0, 2).contiguous().view((batch_size, self._hidden_size * 2, 1))
        q_hn = q_hn.index_select(0, queries_inv_order.cuda())  # batched rows -> reorder batches

        ms = left.bmm(self._mix_matrix.unsqueeze(0).expand(batch_size,
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
        self._var_embedding_layer.weight.grad[0, :] = 0
        # self._embedding_layer.weight.data[0, :] = 0

    @staticmethod
    def softmax(inputs, axis=1):
        input_size = inputs.size()

        trans_input = inputs.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()

        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)

        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)