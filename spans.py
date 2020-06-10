from itertools import compress
import torch
import numpy as np



class TopicSpans:
    def __init__(self, config, data, topic_num, docs_embeddings=None, docs_lengths=None, is_training=True):
        self.config = config
        self.data = data
        self.is_training = is_training

        self.num_tokens = 0

        # origin idx
        self.doc_ids = []
        self.sentence_id = []
        self.origin_start = []
        self.origin_end = []
        self.width = []

        self.text = []
        self.lemma = []

        # bert idx
        self.segment_ids = []
        self.bert_start = []
        self.bert_end = []

        self.labels = []

        # embeddings
        self.start_end_embeddings = []
        self.continuous_embeddings = []

        self.get_all_spans_from_topic(data, topic_num, docs_embeddings, docs_lengths)
        self.create_tensor()


    def set_span_labels(self):
        self.labels = self.data.get_candidate_labels(self.doc_ids, self.origin_start, self.origin_end)



    def create_tensor(self):
        self.doc_ids = np.asarray(self.doc_ids)
        self.sentence_id = torch.tensor(self.sentence_id)
        self.origin_start = torch.tensor(self.origin_start)
        self.origin_end = torch.tensor(self.origin_end)
        self.width = torch.stack(self.width)

        self.segment_ids = torch.tensor(self.segment_ids)
        self.bert_start = torch.tensor(self.bert_start)
        self.bert_end = torch.tensor(self.bert_end)

        if len(self.start_end_embeddings) > 0:
            self.start_end_embeddings = torch.stack(self.start_end_embeddings)

            device = self.start_end_embeddings.device
            self.width = self.width.to(device)


    def get_docs_candidate(self, original_tokens, bert_start_end):
        max_span_width = self.config['max_mention_span']

        num_tokens = len(original_tokens)
        sentences = torch.tensor([x[0] for x in original_tokens])

        # Find all possible spans up to max_span_width in the same sentence
        candidate_starts = torch.tensor(range(num_tokens)).unsqueeze(1).repeat(1, max_span_width)
        candidate_ends = candidate_starts + torch.tensor(range(max_span_width)).unsqueeze(0)
        candidate_start_sentence_indices = sentences.unsqueeze(1).repeat(1, max_span_width)
        padded_sentence_map = torch.cat((sentences, sentences[-1].repeat(max_span_width)))
        candidate_end_sentence_indices = torch.stack(
            list(padded_sentence_map[i:i + max_span_width] for i in range(num_tokens)))
        candidate_mask = (candidate_start_sentence_indices == candidate_end_sentence_indices) * (
                candidate_ends < num_tokens)
        flattened_candidate_mask = candidate_mask.view(-1)
        candidate_starts = candidate_starts.view(-1)[flattened_candidate_mask]
        candidate_ends = candidate_ends.view(-1)[flattened_candidate_mask]
        sentence_span = candidate_start_sentence_indices.view(-1)[flattened_candidate_mask]

        # Original tokens ids
        original_token_ids = torch.tensor([x[1] for x in original_tokens])
        original_candidate_starts = original_token_ids[candidate_starts]
        original_candidate_ends = original_token_ids[candidate_ends]

        # Convert to BERT ids
        bert_candidate_starts = bert_start_end[candidate_starts, 0]
        bert_candidate_ends = bert_start_end[candidate_ends, 1]

        return sentence_span, (original_candidate_starts, original_candidate_ends), \
               (bert_candidate_starts, bert_candidate_ends)



    def get_all_spans_from_topic(self, data, topic_num, docs_embeddings, docs_length):
        # doc names may appear more than once if the doc was splitted into segments
        doc_names = data.topics_list_of_docs[topic_num]

        for i in range(len(doc_names)):
            doc_id = doc_names[i]
            original_tokens = data.topics_origin_tokens[topic_num][i]
            bert_start_end = data.topics_start_end_bert[topic_num][i]
            if self.is_training:  # Filter only the validated sentences according to Cybulska setup
                filt = [x[-1] for x in original_tokens]
                bert_start_end = bert_start_end[filt]
                original_tokens = list(compress(original_tokens, filt))

            if not original_tokens:
                continue

            self.num_tokens += len(original_tokens)
            sentence_span, original_candidates, bert_candidates = self.get_docs_candidate(original_tokens, bert_start_end)
            original_candidate_starts, original_candidate_ends = original_candidates

            #token_text = np.asarray([x[2] for x in original_tokens])



            # update origin idx
            self.doc_ids.extend([doc_id] * len(sentence_span))
            self.sentence_id.extend(sentence_span)
            self.origin_start.extend(original_candidate_starts)
            self.origin_end.extend(original_candidate_ends)
            self.width.extend(original_candidate_ends - original_candidate_starts)

            # update bert idx
            bert_candidate_starts, bert_candidate_ends = bert_candidates
            self.segment_ids.extend([i] * len(sentence_span))
            self.bert_start.extend(bert_candidate_starts)
            self.bert_end.extend(bert_candidate_ends)


            # add span embeddings
            if docs_embeddings is not None:
                doc_embeddings = docs_embeddings[i][torch.tensor(range(docs_length[i]))]  # remove padding
                self.start_end_embeddings.extend(torch.cat((doc_embeddings[bert_candidate_starts],
                                                            doc_embeddings[bert_candidate_ends]), dim=1))
                continuous_tokens_embedding, lengths = self.get_all_token_embedding(doc_embeddings,
                                                                                    bert_candidate_starts,
                                                                                    bert_candidate_ends)
                self.continuous_embeddings.extend(continuous_tokens_embedding)





    def get_all_token_embedding(self, embedding, start, end):
        span_embeddings, length = [], []
        for s, e in zip(start, end):
            indices = torch.tensor(range(s, e + 1))
            span_embeddings.append(embedding[indices])
            length.append(len(indices))
        return span_embeddings, length




    def prune_spans(self, indices):
        # origin idx
        self.doc_ids = self.doc_ids[indices]
        self.sentence_id = self.sentence_id[indices]
        self.origin_start = self.origin_start[indices]
        self.origin_end = self.origin_end[indices]
        self.width = self.width[indices]

        # bert idx
        self.segment_ids = self.segment_ids[indices]
        self.bert_start = self.bert_start[indices]
        self.bert_end = self.bert_end[indices]

        self.labels = self.labels[indices]

        # embeddings
        if len(self.start_end_embeddings) > 0:
            self.start_end_embeddings = self.start_end_embeddings[indices]
            self.continuous_embeddings = [self.continuous_embeddings[x] for x in indices]
