"""Data utilities."""
import os
import random
import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import torch
from torch.autograd import Variable
import time
from src.cuda import CUDA
from src import evaluation

import logging
from utils.log_func import get_log_func
from transformers import OpenAIGPTTokenizer, GPT2Tokenizer#, XLNetTokenizer, TransfoXLTokenizer
from tqdm import tqdm

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

def train_test_split(
    inp_data, out_data=None, test_frac=0.1, valid_frac=0.0, shuffle=True
):
    """ split data into training, testing, validation """
    # TODO set random seed?
    if shuffle:
        perm = np.random.permutation(len(inp_data))
        inp_data = inp_data[perm]
        if out_data is not None:
            out_data = out_data[perm]

    n_test = np.ceil(len(inp_data) * test_frac).astype("int")
    n_valid = np.ceil(len(inp_data) * test_frac).astype("int")
    inp_data_test = inp_data[:n_test]
    inp_data_valid = inp_data[n_test : n_test + n_valid]
    inp_data_train = inp_data[n_test + n_valid :]

    if out_data is not None:
        out_data_test = out_data[:n_test]
        out_data_valid = out_data[n_test : n_test + n_valid]
        out_data_train = out_data[n_test + n_valid :]
        if valid_frac > 0:
            return (
                inp_data_train,
                inp_data_valid,
                inp_data_test,
                out_data_train,
                out_data_valid,
                out_data_test,
            )
        else:
            return inp_data_train, inp_data_test, out_data_train, out_data_test

    if valid_frac > 0:
        return inp_data_train, inp_data_valid, inp_data_test
    else:
        return inp_data_train, inp_data_test

class CorpusSearcher(object):
    """ object that supports searching for similar sequences by tfidf similarity"""
    def __init__(self, query_corpus, key_corpus, value_corpus, vectorizer, make_binary=True):
        self.vectorizer = vectorizer
        self.vectorizer.fit(key_corpus)

        self.query_corpus = query_corpus
        self.key_corpus = key_corpus
        self.value_corpus = value_corpus
        
        # rows = docs, cols = features
        self.key_corpus_matrix = self.vectorizer.transform(key_corpus)
        if make_binary:
            # make binary
            self.key_corpus_matrix = (self.key_corpus_matrix != 0).astype(int)

        
    def most_similar(self, key_idx, n=10):
        """ score the query against the keys and take the corresponding values """
        query = self.query_corpus[key_idx]
        query_vec = self.vectorizer.transform([query])

        scores = np.dot(self.key_corpus_matrix, query_vec.T)
        scores = np.squeeze(scores.toarray())
        scores_indices = zip(scores, range(len(scores)))
        selected = sorted(scores_indices, reverse=True)[:n]

        # use the retrieved i to pick examples from the VALUE corpus
        selected = [
            #(self.query_corpus[i], self.key_corpus[i], self.value_corpus[i], i, score) # useful for debugging 
            (self.value_corpus[i], i, score) 
            for (score, i) in selected
        ]

        return selected

class SalienceCalculator(object):
    """ object that supports the calculation of the saliency of different n-grams across corpii"""
    def __init__(self, corpii, tokenize = None):
        if tokenize is None:
            vectorizers = [CountVectorizer() for _ in corpii]
        else:
            vectorizers = [CountVectorizer(tokenizer=tokenize) for _ in corpii]

        count_matrices = [v.fit_transform(corpus) for v, corpus in zip(vectorizers, corpii)]
        self.vocabs = [v.vocabulary_ for v in vectorizers]
        self.counts = [np.sum(count_matrix, axis=0) for count_matrix in count_matrices]
        self.counts = [np.squeeze(np.asarray(count)) for count in self.counts]

    def max_salience(self, feature, lmbda=0.5):
        # returns index of corpus with highest salience value and this value
        # salience = highest count / 2nd highest count, where highest and 2nd highest counts must be different

        feature_counts = [0.0 for _ in self.counts]
        corpus_indices = [i for i in range(len(self.counts))]
        for idx, (vocab, style_count) in enumerate(zip(self.vocabs, self.counts)):
            if feature in vocab:
                feature_counts[idx] = style_count[vocab[feature]]

        # sort feature counts and corpus indices in tandem
        sort = [(idx, count) for count, idx in sorted(zip(feature_counts, corpus_indices))]
        return sort[-1][0], (sort[-1][1] + lmbda) / (sort[-2][1] + lmbda)

def extract_attributes(line, attribute_vocab, noise='dropout', dropout_prob = 0.1, ngram_range = 5, permutation = 0):
    """ extract attributes from sequnce, either according to noising, word attr, or ngram attr strategy"""

    if noise == 'ngram_attributes':
        # generate all ngrams for the sentence
        grams = []
        for i in range(1, ngram_range):
            try:
                i_grams = [
                    " ".join(gram)
                    for gram in ngrams(line, i) 
                ]
                grams.extend(i_grams)
            except RuntimeError:
                continue

        # filter ngrams by whether they appear in the attribute_vocab
        candidate_markers = [
            (gram, attribute_vocab[gram])
            for gram in grams if gram in attribute_vocab
        ]

        # sort attribute markers by score and prepare for deletion
        content = " ".join(line)
        candidate_markers.sort(key=lambda x: x[1], reverse=True)

        candidate_markers = [marker for (marker, score) in candidate_markers]
        # delete based on highest score first
        attribute_markers = []
        for marker in candidate_markers:
            if marker in content:
                attribute_markers.extend(marker.split())
                content = content.replace(marker, "")
        content = content.split()
        
    elif noise == 'word_attributes':
        content = []
        attribute_markers = []
        for tok in line:
            if tok in attribute_vocab:
                attribute_markers.append(tok)
            else:
                content.append(tok)

    elif noise == 'dropout':
        content = line
        attribute_markers = None
    
    else:
        raise Exception('Noising strategy must be one of "dropout", "word_attributes", or "ngram_attributes"')
    
    # always do noisy masking according to Lample et. al (2017) 
    # word dropout with probability
    for tok in content:
        if np.random.random_sample() < dropout_prob:
            content.remove(tok)

     # permutation with parameter k for content words not dropped
    q = np.random.uniform(0, permutation + 1, len(content))
    q = [q_i + i for q_i, i in zip(q, range(len(q)))]
    content = [tok for _, tok in sorted(zip(q,content))]

    return line, content, attribute_markers

def calculate_ngram_attribute_vocab(input_lines, salience_threshold, ngram_range):
    """ calculates ngram vocabulary for each corpus based on saliency calculations across corpii"""

    def tokenize(text):
        text = text.split()
        grams = []
        for i in range(1, ngram_range):
            i_grams = [
                " ".join(gram)
                for gram in ngrams(text, i)
            ]
            grams.extend(i_grams)
        return grams
    
    # corpii is an iterable of corpus' to calculate attributes over
    def calculate_attribute_markers(corpii):
        sc = SalienceCalculator(prepped_corpii, tokenize)
        attrs = [{} for _ in range(len(corpii))]
        unique_grams = np.array([])
        for corpus in corpii:
            corpus_joined = []
            for sentence in tqdm(corpus):
                for i in range(1, ngram_range):
                    i_grams = ngrams(sentence.split(), i)
                    corpus_joined.extend([
                        " ".join(gram)
                        for gram in i_grams
                    ])
            unique_grams = np.concatenate((unique_grams, np.unique(np.array(corpus_joined))))

        # calculate saliences and return n-gram attribute lists
        for gram in unique_grams:
            salience_index, max_sal = sc.max_salience(gram)
            if max_sal > salience_threshold:
                attrs[salience_index][gram] = max_sal
        return attrs

    prepped_corpii = [[' '.join(line) for line in corpus] for corpus in input_lines]
    return calculate_attribute_markers(prepped_corpii)

def get_tokenizer(encoder = 'gpt2',
    start_token = '<s>',
    stop_token = '</s>',
    pad_token = '<pad>',
    empty_token = '<empty>',
    cache_dir = None
):

    """ gets tokenizer and defines special tokens"""
    # define dicts of tokenizers and tokenizer weights
    tokenizers = {
        'gpt': OpenAIGPTTokenizer, 
        'gpt2': GPT2Tokenizer, 
        # 'xlnet': XLNetTokenizer,
        # 'transformerxl': TransfoXLTokenizer
    }
    tokenizer_weights = {
        'gpt': 'openai-gpt', 
        'gpt2': 'gpt2', 
        # 'xlnet': 'xlnet-base-cased',
        # 'transformerxl': 'transfo-xl-wt103'
    }

    if encoder not in tokenizers.keys():
        raise Exception("Tokenizer must be one of 'gpt', 'gpt2'")#, 'xlnet', 'transformerxl'")
    else:    
        tokenizer = tokenizers[encoder].from_pretrained(
            tokenizer_weights[encoder], 
            cache_dir = cache_dir,
            bos_token = start_token,
            eos_token = stop_token,
            sep_token = stop_token # set sep token to prevent verbose warnings
            # pad_token = pad_token,
            # additional_special_tokens = [empty_token]
        )
        
        # extra token for empty attribute lines in Delete+Retrieve
        special_tokens_dict = {
        #     'bos_token': start_token,
        #     'eos_token': stop_token,
            'pad_token': pad_token,
            'additional_special_tokens': [empty_token]
        }
        tokenizer.add_special_tokens(special_tokens_dict)

    #tokenized_lines = [[str(e) for e in tokenizer.encode(line)] for line in lines]
    return tokenizer

def read_nmt_data(n_styles, config, train_data=None, cache_dir = None):
    """ create dictionary of data objects for each corpus after extract attributes. also init tokenizer """
    """ input lines is list of different style corpus' """

    train_test_string = 'train' if train_data is None else 'test'
    input_lines = [[l.strip().split() for l in open(config['data'][train_test_string][i], 'r')] for i in range(n_styles)]

    # 1. Perform noising
    # A.  do noisy masking according to Lample et. al (2017) - in extract_attributes()
    if config['data']['noise'] == 'dropout':
        attrs = [None for _ in range(n_styles)]

    # B. do masking on attribute vocabulary 
    if config['data']['noise'] == 'word_attributes':

        # try to load attribute vocabs from cache if they exist
        attr_path = os.path.join(cache_dir, 'style_vocabs.pkl')
        if os.path.isfile(attr_path):
            attrs = pickle.load(open(attr_path, "rb"))
        else:
            corpii = [np.array([w for line in lines for w in line]) for lines in input_lines]
            corpii_vocab = np.array([])
            for corpus in corpii:
                corpii_vocab = np.concatenate((corpii_vocab, np.unique(corpus)))
            sc = SalienceCalculator(corpii)

            # extract attributes 
            attrs = [[] for _ in range(n_styles)]
            for tok in corpii_vocab:
                salience_index, max_sal = sc.max_salience(tok)
                if max_sal > config['data']['salience_threshold']:
                    attrs[salience_index].append(tok)
            pickle.dump(attrs, open(attr_path, "wb"))

    # C. do masking on n-gram attribute vocabulary
    elif config['data']['noise'] == 'ngram_attributes':

        # try to load attribute vocabs from cache if they exist
        attr_path = os.path.join(cache_dir, 'style_vocabs.pkl')
        if os.path.isfile(attr_path):
            attrs = pickle.load(open(attr_path, "rb"))
        else:
            attrs = calculate_ngram_attribute_vocab(input_lines, 
                config['data']['salience_threshold'], 
                config['data']['ngram_range'])
            pickle.dump(attrs, open(attr_path, "wb"))
    
    # 2. Extract attributes:
    # data is a list with content and attribute information stored for each style corpus
        # each line in each style corpus has been segmented into content and attribute information according to pre-processing
    # TODO: remove marked attributes from all styles (handle None dropout case)
    #all_attrs = [attr for attr_list in attrs for attr in attr_list]
    data = [list(zip(
        *[extract_attributes(line, attrs[style_idx], config['data']['noise'], config['data']['dropout_prob'],
            config['data']['ngram_range'], config['data']['permutation']) for line in lines]
            )) for style_idx, lines in enumerate(input_lines)]

    # train time: pick attributes that are close to the current (using word distance)      
    # only need to define these distance measurers if using the delete and retrieve model
    if config['model']['model_type'] == 'delete_retrieve':
        if train_data is None:
            dist_measurers = [[CorpusSearcher(
                query_corpus=[' '.join(x) for x in attributes],
                key_corpus=[' '.join(x) for x in attributes],
                value_corpus=[' '.join(x) for x in attributes],
                vectorizer=CountVectorizer(),
                make_binary=True)
                for _ in range(n_styles)]
                for (_, _, attributes) in data]

        # at test time, scan through train content (using tfidf) and retrieve corresponding attributes
        # need to create a test dist measurer for each permutation of styles
        else:
            dist_measurers = [[CorpusSearcher(
                query_corpus=[' '.join(x) for x in test_content],
                key_corpus=[' '.join(x) for x in train_dict['content']],
                value_corpus=[' '.join(x) for x in train_dict['attribute']],
                vectorizer=TfidfVectorizer(),
                make_binary=False)
                for train_dict in train_data]
                for (_, test_content, _) in data]
    else:
        dist_measurers = [None for _ in range(n_styles)]

    # instantiate tokenizer
    tokenizer = get_tokenizer(encoder = config['data']['tokenizer'], cache_dir=cache_dir)

    # create dictionaries of train or test data
    datasets = [{
        'data': lines, 'content': content, 'attribute': attributes,
        'dist_measurer': dist_measurer, 'tokenizer': tokenizer} 
        for (lines, content, attributes), dist_measurer in zip(data, dist_measurers)]

    return datasets

def sample_replace(lines, tokenizer, dist_measurers, sample_rate, corpus_idx):
    """
    replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
    not exactly the same as the paper (words shared instead of jaccaurd during train) but same idea
    only relevant in Delete and Retrieve model where similar sentences substituted
    """

    out = []
    batch_size = len(lines) // len(dist_measurers)
    for i, dist_measurer in enumerate(dist_measurers):
        for j, line in enumerate(lines[i * batch_size:(i+1) * batch_size]):
            if random.random() < sample_rate:
                # top match is the current line
                sims = dist_measurer.most_similar(corpus_idx + j)[1:]
                
                try:
                    line = next( (
                        tgt_attr.split() for tgt_attr, _, _ in sims
                        if set(tgt_attr.split()) != set(line) # and tgt_attr != ''   # TODO -- exclude blanks?
                    ) )
                # all the matches are blanks
                except StopIteration:
                    line = []

            # corner case: special tok for empty sequences 
            if len(line) == 0:
                line.insert(1, tokenizer.additional_special_tokens[0])
            out.append(line)
    return out


def get_minibatch(lines_even, tokenizer, index, batch_size, max_len, sort=False, idx=None,
        dist_measurer=None, sample_rate=0.0):
    """Prepare minibatch. lines_even is a list of lines that contains an even number of samples from each style"""

    # FORCE NO SORTING because we care about the order of outputs
    #   to compare across systems

    lines = [line for lines in lines_even for line in lines[index:index + batch_size]]
    
    # Todo decompose list of dist_measurers
    if dist_measurer is not None:
        lines = sample_replace(lines, tokenizer, dist_measurer, sample_rate, index)

    lines = [
        [tokenizer.bos_token_id] + 
        tokenizer.encode(" ".join(line))[:max_len - 2] + 
        [tokenizer.eos_token_id] for line in lines]

    lens = [len(line) - 1 for line in lines]

    input_lines = [
        line[:-1] +
        [tokenizer.pad_token_id] * (max_len - len(line) + 1)
        for line in lines
    ]

    output_lines = [
        line[1:] +
        [tokenizer.pad_token_id] * (max_len - len(line) + 1)
        for line in lines
    ]

    mask = [
        ([False] * l) + ([True] * (max_len - l))
        for l in lens
    ]

    if sort:
        # sort sequence by descending length
        idx = [x[0] for x in sorted(enumerate(lens), key=lambda x: -x[1])]

    if idx is not None:
        lens = [lens[j] for j in idx]
        input_lines = [input_lines[j] for j in idx]
        output_lines = [output_lines[j] for j in idx]
        mask = [mask[j] for j in idx]

    input_lines = Variable(torch.LongTensor(input_lines))
    output_lines = Variable(torch.LongTensor(output_lines))
    mask = Variable(torch.BoolTensor(mask))
    if CUDA:
        input_lines = input_lines.cuda()
        output_lines = output_lines.cuda()
        mask = mask.cuda()

    return input_lines, output_lines, lens, mask, idx


def back_translation_minibatch(datasets, style_ids, n_styles, config, batch_idx, sample_size, max_len, model, model_type):

    """ get minibatch of sentences for backtranslation. These sentences are generated as discrete sequences
        and thus are not back-propagated through. """
    # get minibatch of inputs, attributes, outputs in other style direction
    input_content, input_aux, _, out_dataset_ordering = minibatch(datasets, style_ids, n_styles, batch_idx, sample_size, max_len, 
        config['model']['model_type'], is_bt = True)

    # decode dataset with greedy, beam search, or top k
    tokenizer = datasets[0]['tokenizer']
    # tgt_pred[i] is list of ids generated by decoding approach
    s = time.time()
    tgt_pred = evaluation.generate_sequences(
        tokenizer,
        model, 
        config,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        input_content,
        input_aux
    )
    log(f'Predicting one BT minibatch took {time.time() - s} seconds', level = 'debug')
    preds = evaluation.ids_to_toks(tgt_pred, tokenizer, sort=False)

    # get minibatch of decoded inputs, attributes, outputs in original style direction
    if config['data']['noise'] == 'dropout':
        attrs = None
    else:
        attr_path = os.path.join('checkpoints', config['data']['vocab_dir'])
        attr_path = os.path.join(attr_path, 'style_vocabs.pkl')
        attrs = pickle.load(open(attr_path, "rb"))

    bt_datasets = []
    for attr_id_orig, attr_id_middle in enumerate(out_dataset_ordering):
        
        # extract attributes from tgt_pred
        batch_len = len(input_content[0]) // len(out_dataset_ordering)
        start_idx = attr_id_orig * batch_len
        stop_idx = (attr_id_orig + 1) * batch_len
        # should we extract all styles or just intermediate for BT approach??
        attribute_vocab = None if attrs == None else attrs[attr_id_middle]
        _, content, _ = list(zip(
            *[extract_attributes(line.split(), attribute_vocab, config['data']['noise'], config['data']['dropout_prob'],
                config['data']['ngram_range'], config['data']['permutation']) for line in preds[start_idx:stop_idx]]
        ))

        # create back translation dataset w/ correct key / value pair
        attributes = datasets[attr_id_orig]['attribute'][batch_idx: batch_idx + batch_len]
        bt_datasets.append({
            'data': [c + a for c, a in zip(content, attributes)], 
            'content': content, 
            'attribute': attributes, 
            'tokenizer': tokenizer,
            'dist_measurer': datasets[attr_id_orig]['dist_measurer']
        })

    # set is_bt false, translate in same direction for evaluation
    # set 0 indexing in minibatch
    bt_minibatch = minibatch(bt_datasets, style_ids, n_styles, 
        batch_idx, sample_size, batch_len, config['model']['model_type'], bt_orig_datasets=datasets)
     
    return bt_minibatch

def minibatch(datasets, style_ids, n_styles, idx, batch_size, max_len, model_type, 
        is_bt = False, is_adv = False, is_test = False, bt_orig_datasets = None):
    """ get minibatch of data - functionality determined by where mb is for backtranslation, train, or test
        datasets is a list of datasets (one of each style) 
        bt_orig_datasets - original datasets before backtranslation (tgt datasets) """
    
    # order lists of input / output datasets depending on whether train, test, BT    
    in_datasets = [datasets[style_id] for style_id in style_ids]
    if bt_orig_datasets is None:
        out_dataset_ordering = []
        out_datasets = datasets
        input_idx = idx
        for i in style_ids:
            if is_bt: # backtranslation: randomly sample different intermediate style
                tgt_idx = random.randint(0, n_styles - 1)
                while tgt_idx == i:
                    tgt_idx = random.randint(0, n_styles - 1)
                out_dataset_ordering.append(tgt_idx)
            elif is_adv: # adversarial: randomly sample intermediate style from list of style_ids
                out_dataset_ordering = style_ids.copy()
                while True:
                    random.shuffle(out_dataset_ordering)
                    if sum([True if i == j else False for i, j in zip(style_ids, out_dataset_ordering)]) >= 1:
                        break
            elif is_test: # test translate to parallel corpus style for accurate evaluation
                if i == 0 or i == 2:
                    tgt_idx = i+1
                else:
                    tgt_idx = i-1
                out_dataset_ordering.append(tgt_idx)
            else: # train, sample style
                tgt_idx = i
                out_dataset_ordering.append(tgt_idx)
    else:
        # handle special BT case opposite direction
        out_dataset_ordering = style_ids
        out_datasets = bt_orig_datasets
        # because bt input dataset always starts at idx 0
        input_idx = 0
    
    in_content = [dataset['content'] for dataset in in_datasets]
    in_data = [dataset['data'] for dataset in in_datasets]
    out_data = [out_datasets[i]['data'] for i in out_dataset_ordering]
    out_attributes = [out_datasets[i]['attribute'] for i in out_dataset_ordering]
    tokenizer = datasets[0]['tokenizer']
    if model_type == 'delete':
        inputs = get_minibatch(
            in_content, tokenizer, input_idx, batch_size, max_len, sort=False)
        outputs = get_minibatch(
            out_data, tokenizer, idx, batch_size, max_len, idx=inputs[-1])

        # true length could be less than batch_size at edge of data
        batch_len = len(inputs[0]) // len(style_ids)
        attribute_ids = [[attribute_id] * batch_len for attribute_id in out_dataset_ordering]
        attribute_ids = [attr_id for id_list in attribute_ids for attr_id in id_list]
        attribute_ids = Variable(torch.unsqueeze(torch.LongTensor(attribute_ids), 1))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        attributes = (attribute_ids, None, None, None, None)

    elif model_type == 'delete_retrieve':
        out_dist_measurers = [dataset['dist_measurer'][i] for dataset, i in zip(in_datasets, out_dataset_ordering)]
        inputs =  get_minibatch(
            in_content, tokenizer, input_idx, batch_size, max_len, sort=False)
        outputs = get_minibatch(
            out_data, tokenizer, idx, batch_size, max_len, idx=inputs[-1])

        if is_test:
            # This dist_measurer has sentence attributes for values, so setting 
            # the sample rate to 1 means the output is always replaced with an
            # attribute. So we're still getting attributes even though
            # the method is being fed content. 
            attributes =  get_minibatch(
                in_content, tokenizer, input_idx, 
                batch_size, max_len, idx=inputs[-1],
                dist_measurer=out_dist_measurers, sample_rate=1.0)
        else:
            attributes =  get_minibatch(
                out_attributes, tokenizer, idx, 
                batch_size, max_len, idx=inputs[-1],
                dist_measurer=out_dist_measurers, sample_rate=0.1)
        attributes = (attributes[0].unsqueeze(-1), attributes[1], attributes[2], attributes[3], attributes[4])

    elif model_type == 'seq2seq':
        # ignore the in/out dataset stuff
        inputs = get_minibatch(
            in_data, tokenizer, input_idx, batch_size, max_len, sort=False)
        outputs = get_minibatch(
            out_data, tokenizer, idx, batch_size, max_len, idx=inputs[-1])
        attributes = (None, None, None, None, None)

    else:
        raise Exception('Unsupported model_type: %s' % model_type)

    # return ds order in back_translation regime, so these attrs can be extracted
    if is_bt or is_adv:
        return inputs, attributes, outputs, out_dataset_ordering
    else:
        return inputs, attributes, outputs

def unsort(arr, idx):
    """unsort a list given idx: a list of each element's 'origin' index pre-sorting
    """
    unsorted_arr = arr[:]
    for i, origin in enumerate(idx):
        unsorted_arr[origin] = arr[i]
    return unsorted_arr



