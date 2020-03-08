import collections
import logging
import os
from itertools import chain
import torch
import random
import numpy as np
from itertools import compress, combinations
import smtplib


def flatten(d, parent_key='', sep=''):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def obj_dict(obj):
    return obj.__dict__



def get_candidate_labels(device, doc_id, start, end, dict_labels1, dict_labels2=None):
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

    return torch.tensor(labels1, device=device), torch.tensor(labels2, device=device)


def get_list_annotated_sentences(annotated_sentences):
    sentences = {}
    for topic, doc, sentence in annotated_sentences:
        if topic not in sentences:
            sentences[topic] = {}
        doc_name = topic + '_' + doc + '.xml'
        if doc_name not in sentences[topic]:
            sentences[topic][doc_name] = []
        sentences[topic][doc_name].append(sentence)
    return sentences


def align_ecb_bert_tokens(ecb_tokens, bert_tokens):
    bert_to_ecb_ids = []
    relative_char_pointer = 0
    ecb_token = None
    ecb_token_id = None

    for bert_token in bert_tokens:
        if relative_char_pointer == 0:
            ecb_token_id, ecb_token, _, _ = ecb_tokens.pop(0)

        bert_token = bert_token.replace("##", "")
        if bert_token == ecb_token:
            bert_to_ecb_ids.append(ecb_token_id)
            relative_char_pointer = 0
        elif ecb_token.find(bert_token) == 0:
            bert_to_ecb_ids.append(ecb_token_id)
            relative_char_pointer = len(bert_token)
            ecb_token = ecb_token[relative_char_pointer:]
        else:
            print("When bert token is longer?")
            raise ValueError((bert_token, ecb_token))

    return bert_to_ecb_ids



def create_logger(config, create_file=True):
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S', format='w')
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    if create_file:
        f_handler = logging.FileHandler(os.path.join(config['save_path'], 'log{}.txt'.format(config['exp_num'])), mode='w')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    logger.propagate = False

    return logger


def separate_docs_into_topics(texts, subtopic=False):
    text_by_topics = {}
    for k, v in texts.items():
        topic_key = k.split('_')[0]
        if subtopic:
            topic_key += '_{}'.format(1 if 'plus' in k else 0)
        if topic_key not in text_by_topics:
            text_by_topics[topic_key] = {}
        text_by_topics[topic_key][k] = v

    return text_by_topics



def get_mentions_by_doc(mentions):
    mentions_by_doc = {}
    for m in list(chain.from_iterable(mentions)):
        if m['doc_id'] not in mentions_by_doc:
            mentions_by_doc[m['doc_id']] = []
        mentions_by_doc[m['doc_id']].append(m)

    return mentions_by_doc


def get_dict_labels(mentions):
    if len(mentions) == 3:
        mentions = list(chain.from_iterable(mentions))
    label_dict = {}
    for m in mentions:
        if m['doc_id'] not in label_dict:
            label_dict[m['doc_id']] = {}
        label_dict[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = m['cluster_id']

    return label_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fix_seed(config):
    torch.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])


def split_doc_into_segments(token_ids, sentence_ids, segment_length=512, with_special_tokens=False):
    segments = [0]
    current_token = 0
    if with_special_tokens:
        segment_length -= 2
    while current_token < len(token_ids):
        end_token = min(len(token_ids) - 1, current_token + segment_length - 1)
        sentence_end = sentence_ids[end_token]
        if end_token != len(token_ids) - 1 and sentence_ids[end_token + 1] == sentence_end:
            while end_token >= current_token and sentence_ids[end_token] == sentence_end:
                end_token -= 1

            if end_token < current_token:
                raise ValueError(token_ids)

        current_token = end_token + 1
        segments.append(current_token)

    return segments




def tokenize_topic(topic, tokenizer):
    list_of_docs = []
    docs_bert_tokens = []
    docs_origin_tokens = []
    docs_start_end_bert = []

    for doc, tokens in topic.items():
        bert_tokens, bert_token_ids = [], []
        ecb_tokens = []
        bert_sentence_ids = []
        start_bert_idx, end_bert_idx = [], []
        alignment = []
        bert_cursor = -1
        for i, token in enumerate(tokens):
            sent_id, token_id, token_text, selected_sentence, continuous_sentence = token
            bert_token = tokenizer.tokenize(token_text)
            if bert_token:
                bert_tokens.extend(bert_token)
                bert_token_ids.extend(tokenizer.convert_tokens_to_ids(bert_token))
                bert_start_index = bert_cursor + 1
                start_bert_idx.append(bert_start_index)
                bert_cursor += len(bert_token)
                bert_end_index = bert_cursor
                end_bert_idx.append(bert_end_index)
                ecb_tokens.append([sent_id, token_id, token_text, selected_sentence])
                bert_sentence_ids.extend([sent_id] * len(bert_token))
                alignment.extend([token_id] * len(bert_token))


        ids = [x[1] for x in ecb_tokens]
        segments = split_doc_into_segments(bert_token_ids, bert_sentence_ids)


        bert_segments, ecb_segments = [], []
        start_end_segment = []
        delta = 0
        for start, end in zip(segments, segments[1:]):
            bert_segments.append(bert_token_ids[start:end])
            start_ecb = ids.index(alignment[start])
            end_ecb = ids.index(alignment[end - 1])
            bert_start = np.array(start_bert_idx[start_ecb:end_ecb + 1]) - delta
            bert_end = np.array(end_bert_idx[start_ecb:end_ecb + 1]) - delta

            if bert_start[0] < 0:
                print('Negative value!!!')
            start_end = np.concatenate((np.expand_dims(bert_start, 1),
                                        np.expand_dims(bert_end, 1)), axis=1)
            start_end_segment.append(start_end)

            ecb_segments.append(ecb_tokens[start_ecb:end_ecb + 1])
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




def pad_and_read_bert(bert_token_ids, bert_model, device):
    length = np.array([len(d) for d in bert_token_ids])
    max_length = max(length)

    if max_length > 512:
        raise ValueError('Error')

    bert_model.eval()

    docs = torch.tensor([doc + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
    attention_masks = torch.tensor([[1] * len(doc) + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
    with torch.no_grad():
        embeddings, _ = bert_model(docs, attention_masks)

    return embeddings, length





def get_all_token_embedding(embedding, start, end):
    span_embeddings, length = [], []
    for s, e in zip(start, end):
        indices = torch.tensor(range(s, e + 1))
        span_embeddings.append(embedding[indices])
        length.append(len(indices))
    return span_embeddings, length



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


def get_all_candidate_from_topic(config, device, doc_names, docs_original_tokens, docs_bert_start_end,
                                 docs_embeddings, docs_length, is_training):
    span_doc, span_sentence, span_origin_start, span_origin_end, span_text = [], [], [], [], []
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
        sentence_span, original_candidates, bert_candidates = get_docs_candidate(original_tokens, bert_start_end, config['max_mention_span'])
        original_candidate_starts, original_candidate_ends = original_candidates
        span_width = (original_candidate_ends - original_candidate_starts).to(device)

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


    # max_length = max(len(v) for v in topic_continuous_embeddings)
    # topic_padded_embeddings = torch.stack(
    #     [torch.cat((emb, padded_vector.repeat(max_length - len(emb), 1)))
    #      for emb in topic_continuous_embeddings]
    # )
    # topic_mask = torch.stack(
    #     [torch.cat((torch.ones(len(emb), device=device), torch.zeros(max_length - len(emb), device=device)))
    #      for emb in topic_continuous_embeddings]
    # )


    return (np.asarray(span_doc), torch.tensor(span_sentence), torch.tensor(span_origin_start), torch.tensor(span_origin_end)), \
           (topic_start_end_embeddings, topic_continuous_embeddings, topic_width), \
           num_tokens


def send_email(user, pwd, recipient, subject, body):

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


