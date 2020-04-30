import torch.nn as nn
import torch
import torch.nn.functional as F



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


class SpanEmbedder(nn.Module):
    def __init__(self, config, bert_hidden_size, device):
        super(SpanEmbedder, self).__init__()
        self.bert_hidden_size = bert_hidden_size
        self.with_width_embedding = config['with_mention_width']
        self.use_head_attention = config['with_head_attention']
        self.device = device
        self.dropout = config['dropout']
        self.padded_vector = torch.zeros(bert_hidden_size, device=device)
        self.self_attention_layer = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(bert_hidden_size, config['hidden_layer']),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config['hidden_layer'], 1)
        )
        self.self_attention_layer.apply(init_weights)
        self.width_feature = nn.Embedding(5, config['embedding_dimension'])


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
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)
            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        return vector



class SpanScorer(nn.Module):
    def __init__(self, config, bert_hidden_size):
        super(SpanScorer, self).__init__()
        self.input_layer = bert_hidden_size * 3
        if config['with_mention_width']:
            self.input_layer += config['embedding_dimension']
        self.mlp = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(self.input_layer, config['hidden_layer']),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config['hidden_layer'], 1)
        )
        self.mlp.apply(init_weights)


    def forward(self, span_embedding):
        return self.mlp(span_embedding)




class SimplePairWiseClassifier(nn.Module):
    def __init__(self, config, bert_hidden_size):
        super(SimplePairWiseClassifier, self).__init__()
        self.input_layer = bert_hidden_size * 3 if config['with_head_attention'] else bert_hidden_size * 2
        if config['with_mention_width']:
            self.input_layer += config['embedding_dimension']
        self.input_layer *= 3
        self.hidden_layer = config['hidden_layer']
        self.pairwise_mlp = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(self.input_layer, self.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )
        self.pairwise_mlp.apply(init_weights)

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))




##################################################################
##### Not used
##################################################################






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
            nn.Dropout(config['dropout']),
            nn.Linear(self.input_layer, self.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            # nn.Linear(self.hidden_layer, self.hidden_layer),
            # nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
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


    def get_span_embedding(self, start_end, continuous_embeddings, width):
        vector = start_end

        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings)
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)

            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        return vector


    def forward(self, start_end, continuous_embeddings, width):
        vector = self.get_span_embedding(start_end, continuous_embeddings, width)
        return vector, self.mlp(vector)




##################################################################


class PairwiseClassifier(nn.Module):
    def __init__(self, config):
        super(PairwiseClassifier, self).__init__()
        self.input_layer = config['hidden_layer'] * 3
        self.dropout = config['dropout']
        self.pairwise_mlp = nn.Sequential(
            nn.Linear(self.input_layer, config['hidden_layer']),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config['hidden_layer'], 1)
        )


    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))












class MentionExtractor(nn.Module):
    def __init__(self, config, bert_hidden_size, device):
        super(MentionExtractor, self).__init__()
        self.bert_hidden_size = bert_hidden_size
        self.hidden_layer = config['hidden_layer']
        self.with_width_embedding = config['with_mention_width']
        self.device = device
        self.padded_vector = torch.zeros(bert_hidden_size, device=device)
        self.self_attention_layer = nn.Sequential(
            nn.Dropout(config['dropout']),
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

        return vector#, self.mlp(vector)



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