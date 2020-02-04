import argparse
import json
import pyhocon
from sklearn.utils import shuffle
from mention_extractor import MentionExtractor
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.optim as optim
from itertools import compress
from evaluator import Evaluation
from utils import *
from copy import deepcopy
import torch.multiprocessing as mp


TRAIN = True
use_avergage = False

parser = argparse.ArgumentParser()
parser.add_argument('--mention_scorer', type=str, default='models/event_mention_scorer_with_attention')
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--config_model_file', type=str, default='config_mention_extractor.json')
parser.add_argument('--train_mention_extractor', type=bool, default=True)
parser.add_argument('--entity_mention_extractor_path', type=str, default='models/entity_mention_extractor_5')
parser.add_argument('--event_mention_extractor_path', type=str, default='models/event_mention_extractor_5')
args = parser.parse_args()



is_training = True




def pad_and_read_bert(bert_token_ids, bert_model, device):
    length = np.array([len(d) for d in bert_token_ids])
    max_length = max(length)

    if max_length > 512:
        raise ValueError('Error')

    docs = torch.tensor([doc + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
    attention_masks = torch.tensor([[1] * len(doc) + [0] * (max_length - len(doc)) for doc in bert_token_ids], device=device)
    with torch.no_grad():
        embeddings, _ = bert_model(docs, attention_masks)

    return embeddings, length





def get_candidate_mention_label(candidate_mentions, doc_gold_mentions):
    gold_mention_spans = {(min(m['tokens_ids']), max(m['tokens_ids'])): m['cluster_id']
                          for m in doc_gold_mentions}
    labels = []
    candidate_starts, candidate_ends = candidate_mentions
    for i, (s, e) in enumerate(zip(candidate_starts.tolist(), candidate_ends.tolist())):
        cluster_id = gold_mention_spans.get((s, e), 0)
        labels.append(cluster_id)

    return torch.tensor(labels)






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




def train_topic_mention_extractor(model, start_end, continuous_embeddings,
                                width, labels, batch_size, criterion, optimizer, device):
    model.train()
    for i in range(0, len(width), batch_size):
        batch_start_end = start_end[i:i+batch_size]
        batch_width = width[i:i+batch_size]
        batch_continuous_embeddings = continuous_embeddings[i:i+batch_size]

        batch_labels = labels[i:i+batch_size]
        optimizer.zero_grad()
        _, preds = model(batch_start_end, batch_continuous_embeddings, batch_width)
        loss = criterion(preds.squeeze(1), batch_labels)
        loss.backward()
        optimizer.step()


def predict_mention_extractor(model, start_end, continuous_embeddings, width):
    model.eval()
    with torch.no_grad():
        _, scores = model(start_end, continuous_embeddings, width)
        scores = scores.squeeze(1)
    return scores




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



def get_all_candidate_from_topic(config, device, doc_names, docs_original_tokens, docs_bert_start_end,
                                 docs_embeddings, docs_length):
    span_doc, span_sentence, span_origin_start, span_origin_end = [], [], [], []
    topic_start_end_embeddings, topic_continuous_embeddings, topic_width = [], [], []
    num_tokens = 0
    padded_vector = torch.zeros(bert_model_hidden_size, device=device)

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
        span_width = (original_candidate_ends.clone().detach() - original_candidate_starts.clone().detach()).to(device)

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








def prepare_data(sp, bert_model, config):
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = sp

    start_end_embeddings = []
    continuous_embeddings = []
    width = []
    num_tokens = 0
    meta_data_0, meta_data_1, meta_data_2, meta_data_3 = [], [], [], []
    dev_bert_model = deepcopy(bert_model).to(config['dev_gpu_num'])

    for t, topic in enumerate(all_topics):
        list_of_docs = topic_list_of_docs[t]
        docs_original_tokens = topic_origin_tokens[t]
        bert_tokens = topic_bert_tokens[t]
        docs_bert_start_end = topic_start_end_bert[t]

        docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, dev_bert_model, config['dev_gpu_num'])
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, config['dev_gpu_num'], list_of_docs, docs_original_tokens, docs_bert_start_end,
            docs_embeddings, docs_length)

        meta_data_0.append(span_meta_data[0])
        meta_data_1.append(span_meta_data[1])
        meta_data_2.append(span_meta_data[2])
        meta_data_3.append(span_meta_data[3])

        num_tokens += num_of_tokens
        topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings

        start_end_embeddings.append(topic_start_end_embeddings)
        continuous_embeddings.append(topic_continuous_embeddings)
        width.append(topic_width)


    return [meta_data_0, meta_data_1, meta_data_2, meta_data_3], \
           [start_end_embeddings, continuous_embeddings, width],\
           num_tokens



def evaluate_model(dev_span_embeddings, model):
    all_scores = []
    start_end_embeddings, continuous_embeddings, width = dev_span_embeddings
    for i in range(len(start_end_embeddings)):
        scores = predict_mention_extractor(model, start_end_embeddings[i], continuous_embeddings[i], width[i])
        all_scores.extend(scores)

    return torch.stack(all_scores)




def get_candidate_labels(doc_id, start, end, dict_labels1, dict_labels2=None):
    labels1, labels2 = torch.zeros(len(doc_id)), torch.zeros(len(doc_id))
    for i, doc in enumerate(doc_id):
        if dict_labels1 and doc in dict_labels1:
            label = dict_labels1[doc].get((start[i].item(), end[i].item()), None)
            if label:
                labels1[i] = label
        if dict_labels2 and doc in dict_labels2:
            label = dict_labels2[doc].get((start[i].item(), end[i].item()), None)
            if label:
                labels2[i] = label

    return labels1, labels2






if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    fix_seed(config)

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    # read and tokenize data
    device = 'cuda:{}'.format(config['gpu_num']) if torch.cuda.is_available() else 'cpu'
    roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model'], add_special_tokens=True)
    sp_raw_text, sp_event_mentions, sp_entity_mentions, sp_tokens = [], [], [], []
    for sp in ['train', 'dev', 'test']:
        logger.info('Processing {} set'.format(sp))
        with open(os.path.join(args.data_folder, sp + '_entities.json'), 'r') as f:
            sp_entity_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '_events.json'), 'r') as f:
            sp_event_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '.json'), 'r') as f:
            ecb_texts = json.load(f)
        ecb_texts_by_topic = separate_docs_into_topics(ecb_texts)
        sp_raw_text.append(ecb_texts_by_topic)
        tokens = tokenize_set(ecb_texts_by_topic, roberta_tokenizer)
        sp_tokens.append(tokens)

    # labels
    logger.info('Get labels')
    event_labels = get_dict_labels(sp_event_mentions)
    entity_labels = get_dict_labels(sp_entity_mentions)





    # Mention extractor configuration
    bert_model = RobertaModel.from_pretrained(config['roberta_model']).to(device)
    bert_model_hidden_size = 768 if 'base' in config['roberta_model'] else 1024
    event_mention_extractor = MentionExtractor(config, bert_model_hidden_size, config['max_mention_span'], config['gpu_num']).to(device)
    event_mention_extractor_clone = MentionExtractor(config, bert_model_hidden_size, config['max_mention_span'],
                                               config['dev_gpu_num']).to(config['dev_gpu_num'])
    event_optimizer = optim.Adam(event_mention_extractor.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


    entity_mention_extractor = MentionExtractor(config, bert_model_hidden_size, config['max_mention_span'], config['gpu_num']).to(device)
    entity_mention_extractor_clone = MentionExtractor(config, bert_model_hidden_size, config['max_mention_span'],
                                                     config['dev_gpu_num']).to(config['dev_gpu_num'])
    entity_optimizer = optim.Adam(entity_mention_extractor.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    logger.info('Number of parameters of mention extractor: {}'.format(count_parameters(event_mention_extractor)))



    if not args.train_mention_extractor:
        event_mention_extractor.load_state_dict(torch.load(args.event_mention_extractor_path))
        entity_mention_extractor.load_state_dict(torch.load(args.entity_mention_extractor_path))




    # Prepare dev data and labels
    dev_candidate_data = prepare_data(sp_tokens[1], bert_model, config)
    dev_doc_id, dev_sentence_id, dev_start, dev_end = dev_candidate_data[0]
    dev_event_labels, dev_entity_labels = [], []
    for i in range(len(dev_doc_id)):
        dev_topic_event_labels, dev_topic_entity_labels = get_candidate_labels(dev_doc_id[i], dev_start[i], dev_end[i], event_labels, entity_labels)
        dev_event_labels.extend(dev_topic_event_labels)
        dev_entity_labels.extend(dev_topic_entity_labels)
    dev_entity_labels = torch.stack(dev_entity_labels).to(config['dev_gpu_num'])
    dev_event_labels = torch.stack(dev_event_labels).to(config['dev_gpu_num'])
    dev_num_tokens = dev_candidate_data[-1]

    dev_entity_mention_labels = torch.zeros(len(dev_entity_labels), device=config['dev_gpu_num'])
    dev_entity_mention_labels[dev_entity_labels.nonzero().squeeze(1)] = 1
    dev_event_mention_labels = torch.zeros(len(dev_event_labels), device=config['dev_gpu_num'])
    dev_event_mention_labels[dev_event_labels.nonzero().squeeze(1)] = 1


    if config['is_event']:
        mention_extractor = event_mention_extractor
        mention_extractor_clone = event_mention_extractor_clone
        optimizer = event_optimizer
        eval_range = [0.2, 0.25, 0.3]
        dev_labels = dev_event_mention_labels
        mention_extractor_path = args.event_mention_extractor_path
        logger.info('Train event mention detection')
    else:
        mention_extractor = entity_mention_extractor
        mention_extractor_clone = entity_mention_extractor_clone
        optimizer = entity_optimizer
        eval_range = [0.2, 0.25, 0.3, 0.4]
        dev_labels = dev_entity_mention_labels
        mention_extractor_path = args.entity_mention_extractor_path
        logger.info('Train entity mention detection')


    training_set = sp_tokens[0]
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = training_set
    logger.info('Number of topics: {}'.format(len(all_topics)))
    max_dev = (0, None)

    for epoch in range(config['epochs']):
        all_scores, all_labels = [], []
        logger.info('Epoch: {}'.format(epoch))

        list_of_topics = shuffle(list(range(len(all_topics))))

        for t in list_of_topics:
            topic = all_topics[t]
            logger.info('Training on topic {}'.format(topic))
            list_of_docs = topic_list_of_docs[t]
            docs_original_tokens = topic_origin_tokens[t]
            bert_tokens = topic_bert_tokens[t]
            docs_bert_start_end = topic_start_end_bert[t]

            docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model, device)
            span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
                config, device, list_of_docs, docs_original_tokens, docs_bert_start_end, docs_embeddings, docs_length)
            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            torch.cuda.empty_cache()

            doc_id, sentence_id, start, end = span_meta_data
            train_event_labels, train_entity_labels = get_candidate_labels(doc_id, start, end, event_labels,
                                                                           entity_labels)

            if args.train_mention_extractor:
                train_labels = train_event_labels if config['is_event'] else train_entity_labels
                mention_labels = torch.zeros(train_labels.shape, device=device)
                mention_labels[train_labels.nonzero().squeeze(1)] = 1

                train_topic_mention_extractor(mention_extractor, topic_start_end_embeddings, topic_continuous_embeddings,
                                              topic_width, mention_labels, config['batch_size'], criterion, optimizer, device)
                torch.cuda.empty_cache()


            else:
                with torch.no_grad():
                    event_span_embeddings, event_span_scores = event_mention_extractor(topic_start_end_embeddings,
                                                                                       topic_continuous_embeddings,
                                                                                       topic_width)
                    entity_span_embeddings, entity_span_scores = entity_mention_extractor(topic_start_end_embeddings,
                                                                                       topic_continuous_embeddings,
                                                                                       topic_width)

                del docs_embeddings, topic_start_end_embeddings, topic_continuous_embeddings, topic_width

                event_span_scores, event_span_indices = torch.topk(event_span_scores.squeeze(1), int(0.3 * num_of_tokens), sorted=False)
                entity_span_scores, entity_span_indices = torch.topk(entity_span_scores.squeeze(1), int(0.4 * num_of_tokens), sorted=False)

                event_span_embeddings = event_span_embeddings[event_span_indices]
                train_event_labels = train_event_labels[event_span_indices]
                entity_span_embeddings = entity_span_embeddings[entity_span_indices]
                train_entity_labels = train_entity_labels[entity_span_indices]

                torch.cuda.empty_cache()
                logger.info('Event mentions: {}'.format(event_span_scores.shape))
                logger.info('Entity mentions: {}'.format(entity_span_scores.shape))
                span_doc, span_sentence, span_origin_start, span_origin_end = span_meta_data
                event_span_doc = span_doc[event_span_indices]
                event_span_sentence = span_doc[event_span_indices]



        if args.train_mention_extractor:
            logger.info('Evaluate on the dev set')

            mention_extractor_clone.load_state_dict(mention_extractor.state_dict())
            all_scores = evaluate_model(dev_candidate_data[1], mention_extractor_clone)
            strict_preds = (all_scores > 0).to(torch.int)

            eval = Evaluation(strict_preds, dev_labels)
            logger.info('Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))

            for k in eval_range:
                s, i = torch.topk(all_scores, int(k * dev_num_tokens), sorted=False)
                rank_preds = torch.zeros(len(all_scores), device=config['dev_gpu_num'])
                rank_preds[i] = 1
                eval = Evaluation(rank_preds, dev_labels)
                recall = eval.get_recall()
                if recall > max_dev[0]:
                    max_dev = (recall, epoch)
                    # torch.save(mention_extractor.state_dict(), mention_extractor_path + '_' + str(config['max_mention_span']))

                logger.info('K = {}, Recall: {}, Precision: {}, F1: {}'.format(k, eval.get_recall(), eval.get_precision(),
                                                                         eval.get_f1()))


    logger.info('Best recall: {}'.format(max_dev[0]))