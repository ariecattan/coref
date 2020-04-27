import numpy as np
import torch
from itertools import compress


def split_doc_into_segments(bert_tokens, sentence_ids, segment_length=512, with_special_tokens=True):
    segments = [0]
    current_token = 0
    if with_special_tokens:
        segment_length -= 2
    while current_token < len(bert_tokens):
        end_token = min(len(bert_tokens) - 1, current_token + segment_length - 1)
        sentence_end = sentence_ids[end_token]
        if end_token != len(bert_tokens) - 1 and sentence_ids[end_token + 1] == sentence_end:
            while end_token >= current_token and sentence_ids[end_token] == sentence_end:
                end_token -= 1

            if end_token < current_token:
                raise ValueError(bert_tokens)

        current_token = end_token + 1
        segments.append(current_token)

    return segments

def tokenize_topic(topic, tokenizer):
    list_of_docs = []
    docs_bert_tokens = []
    docs_origin_tokens = []
    docs_start_end_bert = []


    for doc, tokens in topic.items():
        bert_tokens_ids, bert_sentence_ids = [], []
        ecb_tokens = []
        start_bert_idx, end_bert_idx = [], []
        alignment = []
        bert_cursor = -1
        for i, token in enumerate(tokens):
            sent_id, token_id, token_text, selected_sentence, continuous_sentence = token
            bert_token = tokenizer.encode(token_text)[1:-1]

            if bert_token:
                bert_tokens_ids.extend(bert_token)
                bert_start_index = bert_cursor + 1
                start_bert_idx.append(bert_start_index)
                bert_cursor += len(bert_token)
                bert_end_index = bert_cursor
                end_bert_idx.append(bert_end_index)
                ecb_tokens.append([sent_id, token_id, token_text, selected_sentence])
                bert_sentence_ids.extend([sent_id] * len(bert_token))
                alignment.extend([token_id] * len(bert_token))


        segments = split_doc_into_segments(bert_tokens_ids, bert_sentence_ids)
        ids = [x[1] for x in ecb_tokens]
        bert_segments, ecb_segments, start_end_segment = [], [], []
        delta = 0
        for start, end in zip(segments, segments[1:]):
            start_ecb = ids.index(alignment[start])
            end_ecb = ids.index(alignment[end - 1])
            ecb_segments.append(ecb_tokens[start_ecb:end_ecb + 1])
            bert_ids = tokenizer.encode(' '.join([x[2] for x in ecb_tokens[start_ecb:end_ecb+1]]))[1:-1]
            bert_segments.append(bert_ids)

            bert_start = np.array(start_bert_idx[start_ecb:end_ecb + 1]) - delta
            bert_end = np.array(end_bert_idx[start_ecb:end_ecb + 1]) - delta
            if bert_start[0] < 0:
                print('Negative value!!!')
            start_end = np.concatenate((np.expand_dims(bert_start, 1),
                                        np.expand_dims(bert_end, 1)), axis=1)
            start_end_segment.append(start_end)
            delta = end


        segment_doc = [doc] * (len(segments) - 1)
        docs_start_end_bert.extend(start_end_segment)
        list_of_docs.extend(segment_doc)
        docs_bert_tokens.extend(bert_segments)
        docs_origin_tokens.extend(ecb_segments)

    return list_of_docs, docs_origin_tokens, docs_bert_tokens, docs_start_end_bert


def tokenize_set(raw_text_by_topic, tokenizer):
    all_topics = []
    topic_list_of_docs = []
    topic_origin_tokens, topic_bert_tokens, topic_bert_sentences, topic_start_end_bert = [], [], [], []
    for topic, docs in raw_text_by_topic.items():
        all_topics.append(topic)
        list_of_docs, docs_origin_tokens, docs_bert_tokens, docs_start_end_bert = tokenize_topic(docs, tokenizer)
        topic_list_of_docs.append(list_of_docs)
        topic_origin_tokens.append(docs_origin_tokens)
        topic_bert_tokens.append(docs_bert_tokens)
        topic_start_end_bert.append(docs_start_end_bert)

    return all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert




def pad_and_read_bert(bert_token_ids, bert_model):
    length = np.array([len(d) for d in bert_token_ids])
    max_length = max(length)

    if max_length > 512:
        raise ValueError('Error! Segment too long!')

    device = bert_model.device
    docs = torch.tensor([doc + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
    attention_masks = torch.tensor([[1] * len(doc) + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
    with torch.no_grad():
        embeddings, _ = bert_model(docs, attention_masks)

    return embeddings, length


def get_docs_candidate(original_tokens, bert_start_end, max_span_width):
    num_tokens = len(original_tokens)
    sentences = torch.tensor([x[0] for x in original_tokens])

    # Find all possible spans up to max_span_width in the same sentence
    candidate_starts = torch.tensor(range(num_tokens)).unsqueeze(1).repeat(1, max_span_width)
    candidate_ends = candidate_starts + torch.tensor(range(max_span_width)).unsqueeze(0)
    candidate_start_sentence_indices = sentences.unsqueeze(1).repeat(1, max_span_width)
    padded_sentence_map = torch.cat((sentences, sentences[-1].repeat(max_span_width)))
    candidate_end_sentence_indices = torch.stack(list(padded_sentence_map[i:i + max_span_width] for i in range(num_tokens)))
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





def get_all_token_embedding(embedding, start, end):
    span_embeddings, length = [], []
    for s, e in zip(start, end):
        indices = torch.tensor(range(s, e + 1))
        span_embeddings.append(embedding[indices])
        length.append(len(indices))
    return span_embeddings, length



def get_all_candidate_from_topic(config, doc_names, docs_original_tokens, docs_bert_start_end,
                                 docs_embeddings, docs_length, is_training=True):
    span_doc, span_sentence, span_origin_start, span_origin_end = [], [], [], []
    topic_start_end_embeddings, topic_continuous_embeddings, topic_width = [], [], []
    num_tokens = 0

    for i in range(len(doc_names)):
        doc_id = doc_names[i]
        original_tokens = docs_original_tokens[i]
        bert_start_end = docs_bert_start_end[i]
        if is_training:  # Filter only the validated sentences according to Cybulska setup
            filt = [x[-1] for x in original_tokens]
            bert_start_end = bert_start_end[filt]
            original_tokens = list(compress(original_tokens, filt))

        if not original_tokens:
            continue


        num_tokens += len(original_tokens)
        sentence_span, original_candidates, bert_candidates = get_docs_candidate(original_tokens, bert_start_end,
                                                                                 config['max_mention_span'])
        original_candidate_starts, original_candidate_ends = original_candidates
        span_width = (original_candidate_ends - original_candidate_starts)

        span_doc.extend([doc_id] * len(sentence_span))
        span_sentence.extend(sentence_span)
        span_origin_start.extend(original_candidate_starts)
        span_origin_end.extend(original_candidate_ends)


        bert_candidate_starts, bert_candidate_ends = bert_candidates
        doc_embeddings = docs_embeddings[i][torch.tensor(range(docs_length[i]))]  # remove padding
        continuous_tokens_embedding, lengths = get_all_token_embedding(doc_embeddings, bert_candidate_starts,
                                                                       bert_candidate_ends)
        topic_start_end_embeddings.extend(torch.cat((doc_embeddings[bert_candidate_starts],
                                                     doc_embeddings[bert_candidate_ends]), dim=1))
        topic_width.extend(span_width)
        topic_continuous_embeddings.extend(continuous_tokens_embedding)


    topic_start_end_embeddings = torch.stack(topic_start_end_embeddings)
    topic_width = torch.stack(topic_width)

    return (np.asarray(span_doc), torch.tensor(span_sentence), torch.tensor(span_origin_start), torch.tensor(span_origin_end)), \
           (topic_start_end_embeddings, topic_continuous_embeddings, topic_width), \
           num_tokens



def get_candidate_labels(doc_id, start, end, dict_labels1, dict_labels2=None):
    labels1, labels2 = [0] * len(doc_id), [0] * len(doc_id)
    start = start.tolist()
    end = end.tolist()
    for i, (doc, s, e) in enumerate(zip(doc_id, start, end)):
        if dict_labels1 and doc in dict_labels1:
            label = dict_labels1[doc].get((s, e), None)
            if label:
                labels1[i] = label
        if dict_labels2 and doc in dict_labels2:
            label = dict_labels2[doc].get((s, e), None)
            if label:
                labels2[i] = label

    return torch.tensor(labels1), torch.tensor(labels2)