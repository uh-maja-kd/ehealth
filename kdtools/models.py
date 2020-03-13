import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from kdtools.layers import *

#this is a recycled code, it doesn't make use of our BiLSTM, nor CRF layer
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

        self.bilstmencoder_sent = BiLSTM(input_size, lstm_hidden_size, batch_first=True)
        self.bilstmencoder_stack = BiLSTM(input_size, lstm_hidden_size, batch_first=True)

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

        self.sent_encoder = BiLSTM(embed_size, sent_hidden_size, return_sequence=True, batch_first=True)

        self.origin_encoder = BiLSTM(2*sent_hidden_size, entities_hidden_size, batch_first = True)
        self.destination_encoder = BiLSTM(2*sent_hidden_size, entities_hidden_size, batch_first = True)

        self.origin_dense_hidden = nn.Linear(2*entities_hidden_size, dense_hidden_size)
        self.destination_dense_hidden = nn.Linear(2 * entities_hidden_size, dense_hidden_size)

        self.origin_dropout = nn.Dropout(p=dropout_ratio)
        self.destination_dropout = nn.Dropout(p=dropout_ratio)

        self.dense_output = nn.Linear(2*dense_hidden_size, no_relations)

    def forward(self, X, mask_origin, mask_destination):
        X = self.embedding(X)
        sentence_encoded, _ = self.sent_encoder(X)

        origin_encoded, _ = self.origin_encoder(sentence_encoded * mask_origin)
        destination_encoded, _ = self.destination_encoder(sentence_encoded * mask_destination)

        origin_encoded = torch.tanh(self.origin_dropout(self.origin_dense_hidden(origin_encoded)))
        destination_encoded = torch.tanh(self.destination_dropout(self.destination_dense_hidden(destination_encoded)))
        return F.softmax(self.dense_output(torch.cat((origin_encoded, destination_encoded), dim = 1)), dim = 1)

class BERT_TreeLSTM_BiLSTM_CNN_JointModel(nn.Module):

    def __init__(
        self,
        embedding_size,
        wv,
        bert_size,
        no_postags,
        postag_size,
        no_positions,
        position_size,
        no_chars,
        charencoding_size,
        tree_lstm_hidden_size,
        bilstm_hidden_size,
        local_cnn_channels,
        local_cnn_window_size,
        global_cnn_channels,
        global_cnn_window_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        no_relations
        ):

        super().__init__()

        #INPUT PROCESSING

        #Word Embedding layer
        self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        # self.char_embedding = CharCNN()

        #POS-tag Embedding layer
        self.postag_embedding = nn.Embedding(no_postags, postag_size)

        #Position Embedding layer
        self.position_embedding = nn.Embedding(no_positions, position_size)


        #ENCODING (SHARED PARAMETERS)

        word_rep_size = embedding_size + bert_size + postag_size + position_size + charencoding_size

        #Word-encoding BiLSTM
        self.word_bilstm = BiLSTM(word_rep_size, bilstm_hidden_size//2, return_sequence=True)

        #Word-encoding CNN
        self.word_cnn = nn.Conv1d(word_rep_size, local_cnn_channels, local_cnn_window_size, padding=1)

        #DependencyTree-enconding TreeLSTM
        self.tree_lstm = ChildSumTreeLSTM(word_rep_size, tree_lstm_hidden_size)

        #Global CNN
        self.sentence_cnn = nn.Conv1d(word_rep_size, global_cnn_channels, global_cnn_window_size, padding=1)

        #OUTPUT
        self.dropout = nn.Dropout(dropout_chance)

        sentence_features_size = 2 * (bilstm_hidden_size + local_cnn_channels + tree_lstm_hidden_size) + global_cnn_channels

        #Entity type
        self.entity_type_decoder = nn.Linear(sentence_features_size, no_entity_types)

        #Entites
        self.entities_crf_decoder = CRF(sentence_features_size, no_entity_tags)

        #Relations
        self.relations_decoder = nn.Linear(sentence_features_size, no_relations)

    def forward(self, X):
        # bert_embeddings, word_inputs, char_inputs, postag_inputs, position_inputs, trees, pointed_token_idx = X
        bert_embeddings, word_inputs, char_embeddings, postag_inputs, position_inputs, trees, pointed_token_idx = X
        sent_len = len(trees)

        #obtaining embeddings vectors
        word_embeddings = self.word_embedding(word_inputs)
        # char_embeddings = self.char_embedding(char_inputs)
        postag_embeddings = self.postag_embedding(postag_inputs)
        position_embeddings = self.position_embedding(position_inputs)

        print(
            "bert_embeddings: ", bert_embeddings.shape, "\n",
            "word_embeddings: ", word_embeddings.shape, "\n",
            "char_embeddings: ", char_embeddings.shape, "\n",
            "postag_embeddings: ", postag_embeddings.shape, "\n",
            "position_embeddings: ", position_embeddings.shape, "\n"
        )

        inputs = torch.cat(
            (
                bert_embeddings,
                word_embeddings,
                char_embeddings,
                postag_embeddings,
                position_embeddings
            ), dim=-1)

        print(
            "inputs: ", inputs.shape, "\n"
        )

        #encoding those inputs
        local_bilstm_encoding, _ = self.word_bilstm(inputs)
        local_cnn_encoding = self.word_cnn(inputs.permute(0,2,1)).permute(0,2,1)
        local_deptree_encoding = torch.cat([self.tree_lstm(tree, inputs.squeeze(0))[1] for tree in trees], dim=0).unsqueeze(0)
        global_cnn_encoding = F.max_pool1d(self.sentence_cnn(inputs.permute(0,2,1)), sent_len).permute(0,2,1)

        print(
            "local_bilstm_encoding: ", local_bilstm_encoding.shape, "\n",
            "local_cnn_encoding: ", local_cnn_encoding.shape, "\n",
            "local_deptree_encoding: ", local_deptree_encoding.shape, "\n",
            "global_cnn_encoding: ", global_cnn_encoding.shape, "\n"
        )

        #and putting all of them together
        tokens_info = torch.cat(
            (
                local_bilstm_encoding,
                local_cnn_encoding,
                local_deptree_encoding
            ), dim=-1)

        #vector associated to the highlighted token
        pointed_token_info = tokens_info[:, pointed_token_idx,:].expand(sent_len, -1).unsqueeze(0)

        #expanding global info
        global_info = global_cnn_encoding.expand(-1, sent_len, -1)

        print(
            "tokens_info: ", tokens_info.shape, "\n",
            "pointed_token_info: ", pointed_token_info.shape, "\n",
            "global_info: ", global_info.shape, "\n"
        )

        #finals inputs are a concatenation of token's info, highlighted token's info and global info
        sentence_encoding = torch.cat(
            (
                tokens_info,
                pointed_token_info,
                global_info
            ), dim=-1)
        sentence_encoding = self.dropout(sentence_encoding)

        print(
            "sentence_encoding: ", sentence_encoding.shape, "\n"
        )

        #output entity type
        entitytype_output = F.softmax(self.entity_type_decoder(sentence_encoding), dim = -1)

        #output entities
        _, entities_output = self.entities_crf_decoder(sentence_encoding)

        #output relations
        relations_output = torch.sigmoid(self.relations_decoder(sentence_encoding))

        print(
            "entitytype_output: ", entitytype_output.shape, "\n",
            "entities_output: ", len(entities_output), "\n",
            "relations_output: ", relations_output.shape, "\n"
        )

        return entitytype_output, entities_output, relations_output

