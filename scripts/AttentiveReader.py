import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentiveReader(nn.Module):
    """Attentive Reader"""
    def __init__(self, vocab_size, story_size, question_size, entity_size,
                 embedding_size=100,
                 hidden_size=128,
                 learning_rate=1e-1,
                 dropout=0.2,
                 optimiser='sgd',
                 init_range=0.01,
                 momentum=0.9,
                 embedding_init_range=0.01,
                 max_grad_morm=10,
                 gru_init_std=0.1,
                 use_glove=False,
                 init_mask='../rc_data/init_mask.npy',
                 init_embedding='../rc_data/init_embedding.npy',
                 seed=None):
        super(AttentiveReader, self).__init__()

        # setting attributes
        self._vocab_size = vocab_size
        self._story_size = story_size
        self._question_size = question_size
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate
        self._dropout_rate = dropout
        self._embedding_size = embedding_size
        # self._decay = decay
        self._init_range = init_range
        self._gru_init_std = gru_init_std
        self._max_grad_norm = max_grad_morm
        self._use_glove = use_glove
        self._init_mask = init_mask
        self._init_embedding = init_embedding
        self._embedding_init_range = embedding_init_range

        if seed is not None:
            torch.manual_seed(seed)

        # create layers
        self._embedding_layer = nn.Embedding(vocab_size, embedding_size, 0)
        self._dropout = nn.Dropout(dropout)
        self._recurrent_layer = nn.GRU(embedding_size, hidden_size, 1,
                                       batch_first=True,
                                       bidirectional=True)
        self._question_recurrent_layer = nn.GRU(embedding_size, hidden_size, 1,
                                                batch_first=True,
                                                bidirectional=True)
        # self._ym_layer = nn.Linear(hidden_size * 2, 1, bias=False)
        # self._um_layer = nn.Linear(hidden_size * 2, 1, bias=False)
        # self._rg_layer = nn.Linear(hidden_size * 2, doc_embedding_size, bias=False)
        # self._ug_layer = nn.Linear(hidden_size * 2, doc_embedding_size, bias=False)
        # self._output_layer = nn.Linear(doc_embedding_size, entity_size)
        self._output_layer = nn.Linear(hidden_size * 2, entity_size)
        self._mix_matrix = nn.Parameter(torch.zeros((hidden_size * 2, hidden_size * 2)))

        self.init_weights()
        if optimiser == 'sgd':
            self._optimiser = optim.SGD(self.parameters(), learning_rate, momentum)
        elif optimiser == 'adam':
            self._optimiser = optim.Adam(self.parameters(), learning_rate)
        else:
            raise ValueError("Unknown Optimiser!")

    def init_weights(self):
        init_range = self._init_range
        init_std = self._gru_init_std
        embedding_init_range = self._embedding_init_range

        if self._use_glove:
            self._embedding_layer.weight.data.uniform_(-embedding_init_range, embedding_init_range)
            # TODO: also try smaller init weight
            init_mask = torch.from_numpy(np.load(self._init_mask)).type(torch.FloatTensor)
            init_embedding = torch.from_numpy(np.load(self._init_embedding)).type(torch.FloatTensor)
            self._embedding_layer.weight.data = self._embedding_layer.weight.data * init_mask + init_embedding
            self._embedding_layer.weight.data[0, :] = 0
        else:
            self._embedding_layer.weight.data.uniform_(-init_range, init_range)
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

    def forward(self, stories, questions, stories_lengths, questions_lengths, pack=True):
        batch_size = stories.size()[0]

        s_emb = self._dropout(self._embedding_layer(stories))
        q_emb = self._dropout(self._embedding_layer(questions))

        if pack:
            s_emb = pack_padded_sequence(s_emb, stories_lengths, batch_first=True)

            # gotcha! sum(ByteTensor) does not work! must convert to IntTensor!
            # TODO: flexible cuda on/off
            questions_lengths = Variable(torch.from_numpy(questions_lengths), requires_grad=False)
            _, queries_len_order = torch.sort(questions_lengths, descending=True)  # get order of story length
            _, queries_inv_order = torch.sort(queries_len_order)
            q_emb = q_emb.index_select(0, queries_len_order.cuda())  # sort embeddings in story length order
            queries_len_ordered, _ = torch.sort(questions_lengths, descending=True)  # get story lengths in sorted order
            q_emb = pack_padded_sequence(q_emb, queries_len_ordered.data.numpy(), batch_first=True)

        y_out, _ = self._recurrent_layer(s_emb)  # batch * story_size * 2hidden_size
        _, q_hn = self._question_recurrent_layer(q_emb)  # _, batch * 2hidden_size

        M = 1e2
        if pack:
            y_out, _ = pad_packed_sequence(y_out, batch_first=True)
        padding_mask = (stories <= 0).type(torch.cuda.FloatTensor)[:, :y_out.size(1)]
        # -> batch * max(story_size)
        if not pack:
            y_out = y_out * padding_mask.unsqueeze(2).expand_as(y_out)

        left = y_out.contiguous()
        q_hn = q_hn.permute(1, 0, 2).contiguous().view((batch_size, self._hidden_size * 2, 1))
        if pack:
            q_hn = q_hn.index_select(0, queries_inv_order.cuda())  # batched rows -> reorder batches

        ms = left.bmm(self._mix_matrix.unsqueeze(0).expand(batch_size,
                                                           self._hidden_size * 2,
                                                           self._hidden_size * 2))  # batched matrix
        ms = ms.bmm(q_hn)  # batched [col, col, col, ...] -> batched [scalar, scalar, scalar, ...]

        ms = ms - M * padding_mask.unsqueeze(2)
        # mask is used to force padded words to not be attended to
        ss = F.softmax(ms)  # ss: batch * story_size
        #  TODO: evaluate whether softmax is causing vanishing gradients
        r = torch.sum(y_out * ss.expand_as(y_out), dim=1, keepdim=True)  # batch * 2hidden_size  # TODO: sum/mean

        out = self._output_layer(r.squeeze())  # input to loss function should be logits, not softmax!
        # out = F.softmax(g)
        return out

    def train_on_batch(self, stories, questions, stories_lengths, questions_lengths, answers, pack=True):
        """call net.train() before calling this method!"""
        out = self.forward(stories, questions, stories_lengths, questions_lengths, pack)
        loss = nn.CrossEntropyLoss()(out, answers)
        self.zero_grad()
        loss.backward()
        self._reset_nil_gradients()
        torch.nn.utils.clip_grad_norm(self.parameters(), self._max_grad_norm)
        self._optimiser.step()
        return loss.data, F.softmax(out)

    def predict(self, stories, questions, stories_lengths, questions_lengths, pack=True):
        """call net.eval() before calling this method!"""
        out = self.forward(stories, questions, stories_lengths, questions_lengths, pack)
        return F.softmax(out)

    def _reset_nil_gradients(self):
        self._embedding_layer.weight.grad[0, :] = 0
        # self._embedding_layer.weight.data[0, :] = 0

    @staticmethod
    def softmax(inputs, dim=1):
        input_size = inputs.size()

        trans_input = inputs.transpose(dim, len(input_size) - 1)
        trans_size = trans_input.size()

        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)

        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(dim, len(input_size) - 1)