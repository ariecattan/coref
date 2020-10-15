import numpy as np
import torch
from itertools import compress



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



def get_all_candidate_from_topic(config, data, topic_num, docs_embeddings, docs_length, is_training=True):
    span_doc, span_sentence, span_origin_start, span_origin_end = [], [], [], []
    topic_start_end_embeddings, topic_continuous_embeddings, topic_width = [], [], []
    num_tokens = 0

    doc_names = data.topics_list_of_docs[topic_num]

    for i in range(len(doc_names)):
        doc_id = doc_names[i]
        original_tokens = data.topics_origin_tokens[topic_num][i]
        bert_start_end = data.topics_start_end_bert[topic_num][i]
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



    return (np.asarray(span_doc), torch.tensor(span_sentence), torch.tensor(span_origin_start),
            torch.tensor(span_origin_end)), \
           (topic_start_end_embeddings, topic_continuous_embeddings, topic_width), \
           num_tokens