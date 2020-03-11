import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import kdtools
# from kdtools.layers import BiLSTMEncoder

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, return_sequence = False, **kargs):
        super().__init__()
        self.layer = nn.LSTM(input_size, hidden_size, bidirectional=True)

        self.hidden_size = 2 * self.layer.hidden_size

        self.return_sequence = return_sequence

    def forward(self, input, hx=None):
        output, hidden = self.layer(input, hx)

        hidden_size = self.layer.hidden_size

        left2right = output[:, :, :hidden_size]
        right2left = output[:, :, hidden_size:].flip([1])

        output = torch.cat((left2right, right2left), -1)

        if not self.return_sequence:
            if self.layer.batch_first:
                output = output[-1,:,:]
            else:
                output = output[:,-1,:]

        return output, hidden

class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim, tagset_size, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size+2
        self.START_TAG = tagset_size
        self.STOP_TAG = tagset_size+1

        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        sentence = sentence.view(-1, 1, self.input_dim)
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, no_heads):
        super().__init__()

        self.query_dense = nn.Linear(input_dim, input_dim)
        self.key_dense = nn.Linear(input_dim, input_dim)
        self.value_dense = nn.Linear(input_dim, input_dim)

        self.mh_attention = nn.MultiheadAttention(input_dim, no_heads)

    def forward(self, x):
        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        return self.mh_attention(query, key, value)[0]

class PretrainedEmbedding(nn.Module):

    def __init__(self, wv):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

    def forward(self, x):
        return self.embedding(x)

class EmbeddingBiLSTM_CRF(nn.Module):
    def __init__(self, tagset_size, hidden_dim, wv):
        super().__init__()
        embed_size = len(wv.vectors[0])
        self.embedding = PretrainedEmbedding(wv)
        self.bislstmcrf = BiLSTM_CRF(embed_size, tagset_size, hidden_dim)

    def neg_log_likelihood(self, X, y):
        X = self.embedding(X)
        return self.bislstmcrf.neg_log_likelihood(X, y)

    def forward(self, X):
        return self.bislstmcrf(self.embedding(X))

class EmbeddingAttentionBiLSTM_CRF(nn.Module):
    def __init__(self, tagset_size, hidden_dim, no_heads, wv):
        super().__init__()
        embed_size = len(wv.vectors[0])
        self.wv = wv
        self.embedding = PretrainedEmbedding(wv)
        self.attention = MultiheadAttention(embed_size, no_heads)
        self.bislstmcrf = BiLSTM_CRF(embed_size, tagset_size, hidden_dim)

    def neg_log_likelihood(self, X, y):
        X = self.attention(self.embedding(X))
        return self.bislstmcrf.neg_log_likelihood(X, y)

    def forward(self, X):
        return self.bislstmcrf(self.attention(self.embedding(X)))

class BiLSTMDoubleDenseOracleParser(nn.Module):
    def __init__(self,
        input_size,
        lstm_hidden_size,
        dropout_ratio,
        hidden_dense_size,
        wv,
        actions_no,
        relations_no
    ):
        super().__init__()

        self.wv = wv

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

        self.bilstmencoder_sent = BiLSTMEncoder(input_size, lstm_hidden_size, batch_first=True)
        self.bilstmencoder_stack = BiLSTMEncoder(input_size, lstm_hidden_size, batch_first=True)

        self.dropout_sent = nn.Dropout(p = dropout_ratio)
        self.dropout_stack = nn.Dropout(p = dropout_ratio)

        self.dense_sent = nn.Linear(self.bilstmencoder_sent.hidden_size, hidden_dense_size)
        self.dense_stack = nn.Linear(self.bilstmencoder_stack.hidden_size, hidden_dense_size)

        dense_input_size = 2*hidden_dense_size

        self.action_dense = nn.Linear(dense_input_size, actions_no)
        self.relation_dense = nn.Linear(dense_input_size, relations_no)

    def forward(self, x):
        x0 = self.embedding(x[0])
        x1 = self.embedding(x[1])

        stack_encoded, _ = self.bilstmencoder_stack(x0)
        sent_encoded, _ = self.bilstmencoder_sent(x1)

        stack_encoded = self.dropout_stack(stack_encoded)
        sent_encoded = self.dropout_sent(sent_encoded)

        stack_encoded = torch.tanh(self.dense_stack(stack_encoded))
        sent_encoded = torch.tanh(self.dense_sent(sent_encoded))

        encoded = torch.cat([stack_encoded, sent_encoded], 1)

        action_out = F.softmax(self.action_dense(encoded), 1)
        relation_out = F.softmax(self.relation_dense(encoded), 1)

        return [action_out, relation_out]

class BiLSTMSelectiveRelationClassifier(nn.Module):
    def __init__(self, sent_hidden_size, entities_hidden_size, dense_hidden_size, no_relations, wv, dropout_ratio = 0.2):
        super().__init__()

        embed_size = len(wv.vectors[0])
        self.wv = wv
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

        self.sent_encoder = BiLSTMEncoder(embed_size, sent_hidden_size, return_sequence=True, batch_first=True)

        self.origin_encoder = BiLSTMEncoder(2*sent_hidden_size, entities_hidden_size)
        self.destination_encoder = BiLSTMEncoder(2*sent_hidden_size, entities_hidden_size)

        self.origin_dense_hidden = nn.Linear(2*entities_hidden_size, dense_hidden_size)
        self.destination_dense_hidden = nn.Linear(2 * entities_hidden_size, dense_hidden_size)

        self.origin_dropout = nn.Dropout(p=dropout_ratio)
        self.destination_dropout = nn.Dropout(p=dropout_ratio)

        self.dense_output = nn.Linear(2*dense_hidden_size, no_relations)

    def forward(self, X, mask_origin, mask_destination):
        X = self.embedding(X)
        sentence_encoded, _ = self.sent_encoder(X)

        origin_encoded, _ = self.origin_encoder(sentence_encoded*mask_origin)
        destination_encoded, _ = self.destination_encoder(sentence_encoded * mask_destination)

        origin_encoded = torch.tanh(self.origin_dropout(self.origin_dense_hidden(origin_encoded)))
        destination_encoded = torch.tanh(self.destination_dropout(self.destination_dense_hidden(destination_encoded)))

        return F.softmax(self.dense_output(torch.cat((origin_encoded, destination_encoded), dim = 1)), dim = 1)

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), torch.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state
