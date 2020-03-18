import torch.nn as nn
import torch
import torch.nn.functional as F



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, config, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = config['dropout']
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = config['alpha']
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





class GraphPairwiseClassifier(nn.Module):
    def __init__(self, config, bert_hidden_size):
        super(GraphPairwiseClassifier, self).__init__()
        self.feature_dim = bert_hidden_size * 3
        self.hidden = config['hidden_layer']
        self.head_number = config['head_number']
        self.dropout = config['dropout']

        if config['with_mention_width']:
            self.feature_dim += config['embedding_dimension']

        self.attentions = [GraphAttentionLayer(config, self.feature_dim, config['hidden_layer'], concat=True)
                          for _ in range(self.head_number)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(config, self.hidden * self.head_number, self.hidden, concat=False)




    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
        #return F.log_softmax(x, dim=1)





class SimplePairWiseClassifier(nn.Module):
    def __init__(self, config, bert_hidden_size):
        super(SimplePairWiseClassifier, self).__init__()
        self.input_layer = bert_hidden_size * 3
        if config['with_mention_width']:
            self.input_layer += config['embedding_dimension']
        self.input_layer *= 3
        self.dropout = config['dropout']
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(self.input_layer, config['hidden_layer']),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config['hidden_layer'], 1)
        )

    def forward(self, first, second):
        #first = F.dropout(first, self.dropout)
        #second = F.dropout(second, self.dropout)
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))



class SimpleMentionExtractor(nn.Module):
    def __init__(self, config, bert_hidden_size, device):
        super(SimpleMentionExtractor, self).__init__()
        self.bert_hidden_size = bert_hidden_size
        self.hidden_layer = config['hidden_layer']
        self.with_width_embedding = config['with_mention_width']
        self.device = device
        self.padded_vector = torch.zeros(bert_hidden_size, device=device)
        self.self_attention_layer = nn.Linear(bert_hidden_size, 1)

        self.width_feature = nn.Embedding(5, config['embedding_dimension'])
        self.use_head_attention = config['with_head_attention']
        self.input_layer = 3 * bert_hidden_size if self.use_head_attention else 2 * bert_hidden_size
        if self.with_width_embedding:
            self.input_layer += config['embedding_dimension']


        self.mlp = nn.Sequential(
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1)
        )



    def pad_continous_embeddings(self, continuous_embeddings):
        max_length = max(len(v) for v in continuous_embeddings)
        padded_tokens_embeddings = torch.stack(
            [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
             for emb in continuous_embeddings]
        )
        masks = torch.stack(
            [torch.cat(
                (torch.ones(len(emb), device=self.device), torch.zeros(max_length - len(emb), device=self.device)))
             for emb in continuous_embeddings]
        )
        return padded_tokens_embeddings, masks


    def forward(self, start_end, continuous_embeddings, width):
        vector = start_end

        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings)

            # max_length = max(len(v) for v in continuous_embeddings)
            # padded_tokens_embeddings = torch.stack(
            #     [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
            #      for emb in continuous_embeddings]
            # )
            # masks = torch.stack(
            #     [torch.cat(
            #         (torch.ones(len(emb), device=self.device), torch.zeros(max_length - len(emb), device=self.device)))
            #         for emb in continuous_embeddings]
            # )

            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores, torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)

            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        return vector, self.mlp(vector)




class MentionExtractor(nn.Module):
    def __init__(self, config, bert_hidden_size, device):
        super(MentionExtractor, self).__init__()
        self.bert_hidden_size = bert_hidden_size
        self.hidden_layer = config['hidden_layer']
        self.with_width_embedding = config['with_mention_width']
        self.device = device
        self.padded_vector = torch.zeros(bert_hidden_size, device=device)
        self.self_attention_layer = nn.Sequential(
            nn.Linear(bert_hidden_size, self.hidden_layer),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1)
        ) #nn.Linear(bert_hidden_size, 1)

        self.width_feature = nn.Embedding(5, config['embedding_dimension'])
        self.use_head_attention = config['with_head_attention']
        self.input_layer = 3 * bert_hidden_size if self.use_head_attention else 2 * bert_hidden_size
        if self.with_width_embedding:
            self.input_layer += config['embedding_dimension']


        self.mlp = nn.Sequential(
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1)
        )




    def forward_doc(self, doc_embedding, start, end, width, padded_tokens_embeddings=None, masks=None):
        vector = torch.cat((doc_embedding[start], doc_embedding[end]), dim=1)

        if self.use_head_attention:
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores, torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)

            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)


        width_embedding = self.width_feature(width)
        vector = torch.cat((vector, width_embedding), dim=1)

        return self.mlp(vector)



    def pad_continous_embeddings(self, continuous_embeddings):
        max_length = max(len(v) for v in continuous_embeddings)
        padded_tokens_embeddings = torch.stack(
            [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
             for emb in continuous_embeddings]
        )
        masks = torch.stack(
            [torch.cat(
                (torch.ones(len(emb), device=self.device), torch.zeros(max_length - len(emb), device=self.device)))
             for emb in continuous_embeddings]
        )
        return padded_tokens_embeddings, masks


    def forward(self, start_end, continuous_embeddings, width):
        vector = start_end

        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings)

            # max_length = max(len(v) for v in continuous_embeddings)
            # padded_tokens_embeddings = torch.stack(
            #     [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
            #      for emb in continuous_embeddings]
            # )
            # masks = torch.stack(
            #     [torch.cat(
            #         (torch.ones(len(emb), device=self.device), torch.zeros(max_length - len(emb), device=self.device)))
            #         for emb in continuous_embeddings]
            # )

            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores, torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)

            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        return vector, self.mlp(vector)



class EventMentionExtractor(nn.Module):
    def __init__(self, config):
        super(EventMentionExtractor, self).__init__()
        self.bert_size = 768 if 'base' in config['roberta_model'] else 1024
        self.input_layer = (3 if config["with_avg_span"] else 2) * self.bert_size +\
                           (config["embedding_dimension"] if config["with_mention_width"] else 0)
        self.hidden_layer = 1024
        self.embeddings_width = nn.Embedding(4, config["embedding_dimension"])
        self.mlp = nn.Sequential(
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )


    def forward(self, input):
        score = self.mlp(input)
        return score





class EntityMentionExtractor(nn.Module):
    def __init__(self,  config):
        super(EntityMentionExtractor, self).__init__()
        self.bert_size = 768 if 'base' in config['roberta_model'] else 1024
        self.input_layer = (3 if config["with_avg_mention"] else 2) * self.bert_size + \
                           (config["embedding_dimension"] if config["with_mention_width"] else 0)
        self.hidden_layer = 1024
        self.embeddings_width = nn.Embedding(4, config["embedding_dimension"])
        self.mlp = nn.Sequential(
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )


    def forward(self, input):
            score = self.mlp(input)
            return score