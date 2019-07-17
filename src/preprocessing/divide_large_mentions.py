import os
import argparse
import pickle
import json
import operator
import math
import jsonlines


parser = argparse.ArgumentParser()
parser.add_argument('--mentions_path', type=str, default='data/ecb/mentions/all_entity_gold_mentions.json')
parser.add_argument('--constitiency_tree_path', type=str, default='data/ecb/ecb_constituency_tree')
parser.add_argument('--output_path', type=str, default='data/ecb/mentions')
args = parser.parse_args()


NP_tags = ['NP', 'NML', 'QP', 'NX']
VP_tags = NP_tags


def obj_dict(obj):
    return obj.__dict__



class Mention:
    def __init__(self, coref_chain, topic, doc_id, sent_id, tokens_number, tokens_str, parse_tree=None):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.tokens_number = tokens_number
        self.tokens_str = tokens_str
        self.topic = topic
        self.coref_chain = coref_chain
        self.parse_tree = parse_tree
        self.min_spans = []
        self.is_container_mention = False



    def set_parse_tree(self, all_trees):
        system_sentence_tree = all_trees[self.topic][self.doc_id][self.sent_id]
        ids = list(range(len(system_sentence_tree['word'].split(' '))))
        sentence_tree = TreeNode(system_sentence_tree['word'], ids, system_sentence_tree['nodeType'], system_sentence_tree['children'])
        mention_tree = self.match_subtree(sentence_tree)
        if mention_tree is not None:
            self.parse_tree = mention_tree



    def add_ids_in_subtree(self, subtree):
        if subtree.children and isinstance(subtree.children[0], TreeNode):
            return
        max_id = subtree.ids[0]
        for i, child in enumerate(subtree.children):
            ids = list(range(max_id, max_id + len(child['word'].split(' '))))
            subtree.children[i]['ids'] = ids
            max_id = ids[-1] + 1
        subtree.set_children()




    def match_subtree(self, tree):
        words = tree.word.split(' ')
        if tree.ids[0] ==  self.tokens_number[0] and tree.ids[-1] == self.tokens_number[-1] \
                and words[0] == self.tokens_str.split(' ')[0] and words[-1] == self.tokens_str.split(' ')[-1]:
            return tree

        elif tree.children:
            self.add_ids_in_subtree(tree)
            for child in tree.children:
                if child.ids[0] <= self.tokens_number[0] and child.ids[-1] >= self.tokens_number[-1]:
                    return self.match_subtree(child)



    '''
        This function is for specific cases in which the nodes 
        in the top two level of the mention parse tree do not contain a valid tag.
        E.g., (TOP (S (NP (NP one)(PP of (NP my friends)))))
    '''

    def get_min_span_no_valid_tag(self, root):
        if not root:
            return

        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        accepted_tags = None

        while queue:
            node, depth = queue.pop(0)

            if not accepted_tags:
                if node.nodeType in ['NP', 'NM']:
                    accepted_tags = ['NP', 'NM', 'QP', 'NX']
                elif node.nodeType == 'VP':
                    accepted_tags = ['VP']

            if not node.children and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.nodeType):
                    self.min_spans.append([node.ids, node.word, node.nodeType, node.depth])
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)

            elif (not self.min_spans or depth < terminal_shortest_depth) and node.children and \
                    (depth == 0 or not accepted_tags or node.nodeType in accepted_tags):
                self.add_ids_in_subtree(node)
                for child in node.children:
                    if child.children or (accepted_tags and node.nodeType in accepted_tags):
                        queue.append((child, depth + 1))



    """
       Exluding terminals like comma and paranthesis
    """

    def is_a_valid_terminal_node(self, tag):
        if (any(c.isalpha() for c in tag) or any(c.isdigit() for c in tag) or tag == '%') \
                and (tag != '-LRB-' and tag != '-RRB-') \
                and tag != 'CC' and tag != 'DT' and tag != 'IN':  # not in conjunctions:
            return True
        return False


    def get_valid_node_min_span(self, root, valid_tags, min_spans):
        if not root:
            return

        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        while queue:
            node, depth = queue.pop(0)

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.nodeType):
                    min_spans.append([node.ids, node.word, node.nodeType, node.depth])
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)

            elif (not min_spans or depth < terminal_shortest_depth) and node.children and \
                    (depth == 0 or not valid_tags or node.nodeType[0:2] in valid_tags):
                self.add_ids_in_subtree(node)
                for child in node.children:
                    if not child.isTerminal or (valid_tags and node.nodeType[0:2] in valid_tags):
                        queue.append((child, depth + 1))


    def get_top_level_phrases(self, root, valid_tags):
        terminal_shortest_depth = float('inf')
        top_level_valid_phrases = []

        if root and root.isTerminal and self.is_a_valid_terminal_node(root.nodeType):
            self.min_spans.append([root.ids, root.word, root.nodeType, root.depth])

        elif root and root.children:
            self.add_ids_in_subtree(root)
            for node in root.children:
                if node:
                    if node.isTerminal and self.is_a_valid_terminal_node(node.nodeType):
                        self.min_spans.append([node.ids, node.word, node.nodeType, node.depth])
            if not self.min_spans:
                for node in root.children:
                    if not node.isTerminal and node.nodeType in valid_tags:
                        top_level_valid_phrases.append(node)

        return top_level_valid_phrases

    def get_valid_tags(self, root):
        valid_tags = None
        NP_tags = ['NP', 'NM', 'QP', 'NX']
        VP_tags = ['VP']

        if root.nodeType[0:2] == 'VP':
            valid_tags = VP_tags
        elif root.nodeType[0:2] in ['NP', 'NM']:
            valid_tags = NP_tags
        else:
            if root.children:  ## If none of the first level nodes are either NP or VP, examines their children for valid mention tags
                all_tags = []
                for node in root.children:
                    all_tags.append(node['nodeType'][0:2])
                if 'NP' in all_tags or 'NM' in all_tags:
                    valid_tags = NP_tags
                elif 'VP' in all_tags:
                    valid_tags = VP_tags
                else:
                    valid_tags = NP_tags

        return valid_tags



    def set_min_span(self):
        if not self.parse_tree:
            print('The parse tree should be set before extracting minimum spans')
            return NotImplemented

        root = self.parse_tree

        if not root:
            return

        terminal_shortest_depth = math.inf
        queue = [(root, 0)]

        valid_tags = self.get_valid_tags(root)

        top_level_valid_phrases = self.get_top_level_phrases(root, valid_tags)

        if self.min_spans:
            return

        '''
        In structures like conjunctions the minimum span is determined independently
        for each of the top-level NPs
        '''

        if top_level_valid_phrases:
            for node in top_level_valid_phrases:
                self.get_valid_node_min_span(node, valid_tags, self.min_spans)

        else:
            self.get_min_span_no_valid_tag(root)

        """
        If there was no valid minimum span due to parsing errors return the whole span
        """
        '''
        if len(self.min_spans) == 0:
            self.min_spans.update([(word, index) for index, word in enumerate(self.words)])
        '''




class NestedMentions:
    def group_nested_mentions(self, mentions):
        nested_mentions = {}
        for mention in mentions:
            doc_id, sent_id = mention.doc_id, mention.sent_id
            nested_mentions[doc_id] = nested_mentions.get(doc_id, {})
            nested_mentions[doc_id][sent_id] = nested_mentions[doc_id].get(sent_id, [])
            nested_mentions[doc_id][sent_id].append(mention)

        return nested_mentions

    def check_nested_mentions(self, large, small):
        if large.coref_chain == small.coref_chain and \
                large.tokens_number[0] <= small.tokens_number[0] and \
                large.tokens_number[-1] >= small.tokens_number[-1]:
            return True

        return False

    def extract_largest_mention(self, sentence_mentions):
        larger_mentions = []
        dic_length = {}
        for mention in sentence_mentions:
            length = len(mention.tokens_number)
            dic_length[length] = dic_length.get(length, [])
            dic_length[length].append(mention)

        all_mentions = []
        for length, mentions in sorted(dic_length.items(), key=operator.itemgetter(0)):
            all_mentions.extend(mentions)

        while all_mentions:
            large = all_mentions[-1]
            flag = True
            for m in range(len(all_mentions) - 2, 0, -1):
                if self.check_nested_mentions(large, all_mentions[m]):
                    larger_mentions.append(large)
                    all_mentions.pop()
                    flag = False
                    break

            if flag:
                all_mentions.pop()

        return larger_mentions


    def set_supplement_mentions(self, mentions):
        grouped_mentions = self.group_nested_mentions(mentions)
        for doc, sentences in grouped_mentions.items():
            for sentence, sentence_mentions in sentences.items():
                largest_mentions = self.extract_largest_mention(sentence_mentions)
                for m in largest_mentions:
                    m.is_container_mention = True




class TreeNode:
    def __init__(self, word, ids, nodeType, children, depth=0):
        self.word = word
        self.ids = ids
        self.nodeType = nodeType
        self.children = children
        self.depth = depth
        self.isTerminal = self.children is None


    def set_children(self):
        children_nodes = []
        for child in self.children:
            node = TreeNode(child['word'], child['ids'], child['nodeType'], child.get('children', None), self.depth + 1)
            children_nodes.append(node)

        self.children = children_nodes



def get_mention(mentions, doc_id, sent_id, start_token_id, end_token_id):
    for mention in mentions:
        if mention.doc_id == doc_id and mention.sent_id == sent_id and mention.tokens_number[0] == start_token_id\
            and mention.tokens_number[-1] == end_token_id:
            return mention
    return None


def main():
    return NotImplemented





if __name__ == '__main__':
    constituency_trees = {}
    for file in os.listdir(args.constitiency_tree_path):
        if not file.endswith('txt'):
            with open(os.path.join(args.constitiency_tree_path, file), 'rb') as f:
                constituency_trees[file] = pickle.load(f)

    with open(args.mentions_path, 'r') as f:
        mentions_raw = json.load(f)

    mentions = []
    for m in mentions_raw:
        mention = Mention(m['coref_chain'], m['topic'], m['doc_id'], m['sent_id'], m['tokens_number'], m['tokens_str'])
        mention.set_parse_tree(constituency_trees)
        if mention.parse_tree:
            mention.set_min_span()
        mentions.append(mention)
    nested_mentions = NestedMentions()
    nested_mentions.set_supplement_mentions(mentions)
    clean_mentions = [m for m in mentions if not m.is_container_mention]
    print('Container mentions to be deleted: {}/{} - {}'.format(len(mentions) - len(clean_mentions), len(mentions), (len(mentions) - len(clean_mentions)) / len(mentions)))
    missing_subtrees = [m for m in clean_mentions if not m.parse_tree]
    num_matched_subtrees = len(clean_mentions) - len(missing_subtrees)
    print('Number of matched mention subtrees: {}/{} - {}'.format(num_matched_subtrees, len(clean_mentions), num_matched_subtrees/ len(clean_mentions)))
    all_min_spans = [m for m in clean_mentions if m.min_spans]
    print('Number of mininum span: {}/{} - {}'.format(len(all_min_spans), num_matched_subtrees, len(all_min_spans)/ num_matched_subtrees))


    with open(os.path.join(args.output_path, 'missing_subtrees.json'), 'w') as f:
        json.dump(missing_subtrees, f, default=obj_dict, indent=4, sort_keys=True)

    with open(os.path.join(args.output_path, 'clean_mentions.json'), 'w') as f:
        json.dump(clean_mentions, f, default=obj_dict, indent=4, sort_keys=True)

    with jsonlines.open(os.path.join(args.output_path, 'compare_min_to_original_span.jsonl'), 'w') as f:
        for mention in all_min_spans:
            origin = mention.tokens_str
            mina =  " ".join([token_str for token_num, token_str, _, _  in mention.min_spans])
            m = {
                "origin": origin,
                "mina": mina
                #"doc_id": mention.doc_id,
                #"sent_id": mention.sent_id
            }
            if origin != mina:
                f.write(m)


