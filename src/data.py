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
from pytorch_transformers import OpenAIGPTTokenizer, GPT2Tokenizer#, XLNetTokenizer, TransfoXLTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

def train_test_split(
    inp_data, out_data=None, test_frac=0.1, valid_frac=0.0, shuffle=True
):
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
    def __init__(self, pre_corpus, post_corpus, tokenize = None):
        if tokenize is None:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = CountVectorizer(tokenizer=tokenize)

        pre_count_matrix = self.vectorizer.fit_transform(pre_corpus)
        self.pre_vocab = self.vectorizer.vocabulary_
        self.pre_counts = np.sum(pre_count_matrix, axis=0)
        self.pre_counts = np.squeeze(np.asarray(self.pre_counts))

        post_count_matrix = self.vectorizer.fit_transform(post_corpus)
        self.post_vocab = self.vectorizer.vocabulary_
        self.post_counts = np.sum(post_count_matrix, axis=0)
        self.post_counts = np.squeeze(np.asarray(self.post_counts))

    def salience(self, feature, attribute='pre', lmbda=0.5):
        assert attribute in ['pre', 'post']
        if feature not in self.pre_vocab:
            pre_count = 0.0
        else:
            pre_count = self.pre_counts[self.pre_vocab[feature]]

        if feature not in self.post_vocab:
            post_count = 0.0
        else:
            post_count = self.post_counts[self.post_vocab[feature]]

        if attribute == 'pre':
            return (pre_count + lmbda) / (post_count + lmbda)
        else:
            return (post_count + lmbda) / (pre_count + lmbda)


# def build_vocab_maps(vocab_file):
#     assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file
#     unk = '<unk>'
#     pad = '<pad>'
#     sos = '<s>'
#     eos = '</s>'

#     lines = [x.strip() for x in open(vocab_file)]

#     assert lines[0] == unk and lines[1] == pad and lines[2] == sos and lines[3] == eos, \
#         "The first words in %s are not %s, %s, %s, %s" % (vocab_file, unk, pad, sos, eos)

#     tok_to_id = {}
#     id_to_tok = {}
#     for i, vi in enumerate(lines):
#         tok_to_id[vi] = i
#         id_to_tok[i] = vi

#     # Extra vocab item for empty attribute lines
#     empty_tok_idx =  len(id_to_tok)
#     tok_to_id['<empty>'] = empty_tok_idx
#     id_to_tok[empty_tok_idx] = '<empty>'

#     return tok_to_id, id_to_tok


def extract_attributes(line, attribute_vocab, noise='dropout', dropout_prob = 0.1, ngram_range = 5, permutation = 0):

    # do noisy masking according to Lample et. al (2017)
    if noise == 'dropout':
        content = []
        attribute_markers = []
        # word dropout with probability
        for tok in line:
            if np.random.random_sample() > dropout_prob:
                content.append(tok)
            else:
                # TODO: maybe just dropout
                # we set non content tokens as attribute tokens, allows replacement in noising model
                attribute_markers.append(tok)

    elif noise == 'ngram_attributes':
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
    else:
        raise Exception('Noising strategy must be one of "dropout", "word_attributes", or "ngram_attributes"')
    
    # permutation with parameter k for content words not dropped
    q = np.random.uniform(0, permutation + 1, len(content))
    q = [q_i + i for q_i, i in zip(q, range(len(q)))]
    content = [tok for _, tok in sorted(zip(q,content))]

    return line, content, attribute_markers

def calculate_ngram_attribute_vocab(tokenized_src_lines, tokenized_tgt_lines, salience_threshold, ngram_range):

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
    
    # corpii is a tuple of corpus' to calculate attributes over
    def calculate_attribute_markers(corpii):
        pre_attr = {}
        post_attr = {}
        for corpus in corpii:
            for sentence in tqdm(corpus):
                for i in range(1, ngram_range):
                    i_grams = ngrams(sentence.split(), i)
                    joined = [
                        " ".join(gram)
                        for gram in i_grams
                    ]
                    for gram in joined:
                        negative_salience = sc.salience(gram, attribute='pre')
                        positive_salience = sc.salience(gram, attribute='post')
                        if max(negative_salience, positive_salience) > salience_threshold:
                            pre_attr[gram] = negative_salience
                            post_attr[gram] = positive_salience
        return pre_attr, post_attr

    prepped_src = [' '.join(line) for line in tokenized_src_lines]
    prepped_tgt = [' '.join(line) for line in tokenized_tgt_lines]
    sc = SalienceCalculator(prepped_src, prepped_tgt, tokenize)
    return calculate_attribute_markers((prepped_src, prepped_tgt))

def get_padding_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def get_start_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

def get_stop_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

def get_empty_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])

def get_start_token(tokenizer):
    return tokenizer.decode(tokenizer.encode(tokenizer.bos_token))

def get_stop_token(tokenizer):
    return tokenizer.decode(tokenizer.encode(tokenizer.eos_token))

def get_tokenizer(encoder = 'gpt2',
    start_token = '<s>',
    stop_token = '</s>',
    pad_token = '<pad>',
    empty_token = '<empty>',
    cache_dir = None
):

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

def read_nmt_data(src_lines, tgt_lines, config, train_src=None, train_tgt=None, cache_dir = None):

    # 1. Tokenize raw text data
    # tokenized_src_lines, tokenizer = encode_text_data(src_lines, 
    #     encoder = config['data']['tokenizer'], 
    #     cache_dir=cache_dir
    # )
    # tokenized_tgt_lines, _ = encode_text_data(tgt_lines, 
    #     encoder = config['data']['tokenizer'], 
    #     cache_dir=cache_dir
    # )
    # tokenized_src_corpus = [w for line in tokenized_src_lines for w in line]
    # tokenized_tgt_corpus = [w for line in tokenized_tgt_lines for w in line]
    # tokenized_vocab = np.unique(np.array(tokenized_src_corpus + tokenized_tgt_corpus))

    # 1. Perform noising
    # A.  do noisy masking according to Lample et. al (2017) - in extract_attributes()
    if config['data']['noise'] == 'dropout':
        pre_attr = post_attr = None

    # B. do masking on attribute vocabulary 
    if config['data']['noise'] == 'word_attributes':

        # try to load attribute vocabs from cache if they exist
        pre_attr_path = os.path.join(cache_dir, 'pre_attribute_vocab.pkl')
        post_attr_path = os.path.join(cache_dir, 'post_attribute_vocab.pkl')
        if os.path.isfile(pre_attr_path) and os.path.isfile(post_attr_path):
            pre_attr = pickle.load(open(pre_attr_path, "rb"))
            post_attr = pickle.load(open(post_attr_path, "rb"))
        else:
            src_corpus = [w for line in src_lines for w in line]
            tgt_corpus = [w for line in tgt_lines for w in line]
            corpus_vocab = np.unique(np.array(src_corpus + tgt_corpus))
            sc = SalienceCalculator(src_corpus, tgt_corpus)
            # extract attributes 
            pre_attr = post_attr = set([tok for tok in corpus_vocab if max(sc.salience(tok, attribute='pre'), sc.salience(tok, attribute='post')) > config['data']['salience_threshold']])
            pickle.dump(pre_attr, open(pre_attr_path, "wb"))
            pickle.dump(post_attr, open(post_attr_path, "wb"))

    # C. do masking on n-gram attribute vocabulary
    elif config['data']['noise'] == 'ngram_attributes':

        # try to load attribute vocabs from cache if they exist
        pre_attr_path = os.path.join(cache_dir, 'pre_attribute_vocab.pkl')
        post_attr_path = os.path.join(cache_dir, 'post_attribute_vocab.pkl')
        if os.path.isfile(pre_attr_path) and os.path.isfile(post_attr_path):
            pre_attr = pickle.load(open(pre_attr_path, "rb"))
            post_attr = pickle.load(open(post_attr_path, "rb"))
        else:
            pre_attr, post_attr = calculate_ngram_attribute_vocab(src_lines, 
                tgt_lines,
                config['data']['salience_threshold'], 
                config['data']['ngram_range'])
            pickle.dump(pre_attr, open(pre_attr_path, "wb"))
            pickle.dump(post_attr, open(post_attr_path, "wb"))
    
    # 2. Extract attributes:
    src_lines, src_content, src_attribute = list(zip(
        *[extract_attributes(line, pre_attr, config['data']['noise'], config['data']['dropout_prob'],
            config['data']['ngram_range'], config['data']['permutation']) for line in src_lines]
    ))
    tgt_lines, tgt_content, tgt_attribute = list(zip(
        *[extract_attributes(line, post_attr, config['data']['noise'], config['data']['dropout_prob'],
            config['data']['ngram_range'], config['data']['permutation']) for line in tgt_lines]
    ))

    # train time: just pick attributes that are close to the current (using word distance)
    # we never need to do the TFIDF thing with the source because 
    # test time is strictly in the src => tgt direction. 
    # But we still both src and tgt dist measurers because training is bidirectional
    #  (i.e., we're autoencoding src and tgt sentences during training)        
    if train_src is None or train_tgt is None:
        src_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in src_attribute],
            key_corpus=[' '.join(x) for x in src_attribute],
            value_corpus=[' '.join(x) for x in src_attribute],
            vectorizer=CountVectorizer(),
            make_binary=True
        )
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in tgt_attribute],
            key_corpus=[' '.join(x) for x in tgt_attribute],
            value_corpus=[' '.join(x) for x in tgt_attribute],
            vectorizer=CountVectorizer(),
            make_binary=True
        )
    # at test time, scan through train content (using tfidf) and retrieve corresponding attributes
    else:
        src_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in tgt_content],
            key_corpus=[' '.join(x) for x in train_src['content']],
            value_corpus=[' '.join(x) for x in train_src['attribute']],
            vectorizer=TfidfVectorizer(),
            make_binary=False
        )
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in src_content],
            key_corpus=[' '.join(x) for x in train_tgt['content']],
            value_corpus=[' '.join(x) for x in train_tgt['attribute']],
            vectorizer=TfidfVectorizer(),
            make_binary=False
        )
    
    # instantiate tokenizer
    tokenizer = get_tokenizer(encoder = config['data']['tokenizer'], cache_dir=cache_dir)

    # create dictionaries of src and tgt data
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'dist_measurer': src_dist_measurer, 'tokenizer': tokenizer    
    }
    tgt = {
        'data': tgt_lines, 'content': tgt_content, 'attribute': tgt_attribute,
        'dist_measurer': tgt_dist_measurer, 'tokenizer': tokenizer
    }
    return src, tgt

def sample_replace(lines, tokenizer, dist_measurer, sample_rate, corpus_idx):
    """
    replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
    not exactly the same as the paper (words shared instead of jaccaurd during train) but same idea
    """

    out = [None for _ in range(len(lines))]
    #replace_count = 0
    for i, line in enumerate(lines):
        if random.random() < sample_rate:
            # top match is the current line
            sims = dist_measurer.most_similar(corpus_idx + i)[1:]
            
            try:
                line = next( (
                    tgt_attr.split() for tgt_attr, _, _ in sims
                    if set(tgt_attr.split()) != set(line[1:-1]) # and tgt_attr != ''   # TODO -- exclude blanks?
                ) )
            # all the matches are blanks
            except StopIteration:
                line = []
            #line = line

        # corner case: special tok for empty sequences (just start/end tok)
        if len(line) == 0:
            #replace_count += 1
            line.insert(1, tokenizer.additional_special_tokens[0])
        out[i] = line
    #print(f'REPLACE_COUNT: {replace_count}')
    return out


def get_minibatch(lines, tokenizer, index, batch_size, max_len, sort=False, idx=None,
        dist_measurer=None, sample_rate=0.0):
    """Prepare minibatch."""
    # FORCE NO SORTING because we care about the order of outputs
    #   to compare across systems

    lines = [line[:max_len] for line in lines[index:index + batch_size]]

    if dist_measurer is not None:
        lines = sample_replace(lines, tokenizer, dist_measurer, sample_rate, index)

    lines = [tokenizer.encode(tokenizer.bos_token + " ".join(line) + tokenizer.eos_token) for line in lines]
    lens = [len(line) - 1 for line in lines]
    max_len = max(lens)
    
    input_lines = [
        line[:-1] +
        [get_padding_id(tokenizer)] * (max_len - len(line) + 1)
        for line in lines
    ]

    output_lines = [
        line[1:] +
        [get_padding_id(tokenizer)] * (max_len - len(line) + 1)
        for line in lines
    ]

    mask = [
        ([True] * l) + ([False] * (max_len - l))
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
    mask = Variable(torch.FloatTensor(mask))

    if CUDA:
        input_lines = input_lines.cuda()
        output_lines = output_lines.cuda()
        mask = mask.cuda()

    return input_lines, output_lines, lens, mask, idx


def back_translation_minibatch(src, tgt, config, idx, batch_size, max_len, model, model_type, use_src = True):

    # get minibatch of inputs, attributes, outputs in other style direction
    input_content, input_aux, output = minibatch(
        src, tgt, idx, batch_size, max_len, config['model']['model_type'], use_src = use_src, is_bt = True)

    # decode dataset with greedy, beam search, or top k
    tokenizer = src['tokenizer']
    # tgt_pred[i] is list of ids generated by decoding approach
    s = time.time()
    tgt_pred = evaluation.generate_sequences(
        tokenizer,
        model, 
        config,
        get_start_id(tokenizer),
        get_stop_id(tokenizer),
        input_content,
        input_aux,
        output
    )
    logging.info(f'Predicting one BT minibatch took {time.time() - s} seconds')
    preds = evaluation.ids_to_toks(tgt_pred, tokenizer, sort=False)

    # get minibatch of decoded inputs, attributes, outputs in original style direction
    # extract attributes from tgt_pred
    if config['data']['noise'] == 'dropout':
        attr = None
    else:
        join_path = 'post_attribute_vocab.pkl' if use_src else 'pre_attribute_vocab.pkl'
        attr_path = os.path.join('checkpoints', config['data']['vocab_dir'])
        attr_path = os.path.join(attr_path, join_path)
        attr = pickle.load(open(attr_path, "rb"))
    _, content, _ = list(zip(
        *[extract_attributes(line.split(), attr, config['data']['noise'], config['data']['dropout_prob'],
            config['data']['ngram_range'], config['data']['permutation']) for line in preds]
    ))
    # print(f'content: {content[0]}')
    # print(f"attr: {src['attribute'][0]}")

    # create back translation dictionary w/ correct key / value pairs
    dist_measurer = src['dist_measurer'] if use_src else tgt['dist_measurer']
    bt_src = {
        'data': [c + a for c, a in zip(content, src['attribute'])], 
        'content': content, 
        'attribute': src['attribute'], 
        'tokenizer': tokenizer,
        'dist_measurer': dist_measurer
    }
    # set is_bt false, translate in same direction for evaluation
    bt_minibatch = minibatch(bt_src, tgt, 0, batch_size, max_len, config['model']['model_type'], use_src = use_src)
    return bt_minibatch

def minibatch(src, tgt, idx, batch_size, max_len, model_type, use_src = True, is_test=False, is_bt = False):

    if not is_test:
        #use_src = random.random() < 0.5
        in_dataset = src if use_src else tgt
        
        # flip attribute ids if generating backtranslation minibatch
        if is_bt:
            attribute_id = 1 if use_src else 0
            out_dataset = tgt if use_src else src
        else:
            attribute_id = 0 if use_src else 1
            out_dataset = in_dataset

    else:
        in_dataset = src
        out_dataset = tgt
        attribute_id = 1

    if model_type == 'delete':
        inputs = get_minibatch(
            in_dataset['content'], in_dataset['tokenizer'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tokenizer'], idx, batch_size, max_len, idx=inputs[-1])

        # true length could be less than batch_size at edge of data
        batch_len = len(outputs[0])
        attribute_ids = [attribute_id for _ in range(batch_len)]
        attribute_ids = Variable(torch.LongTensor(attribute_ids))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        attributes = (attribute_ids, None, None, None, None)

    elif model_type == 'delete_retrieve':
        inputs =  get_minibatch(
            in_dataset['content'], in_dataset['tokenizer'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tokenizer'], idx, batch_size, max_len, idx=inputs[-1])

        if is_test:
            # This dist_measurer has sentence attributes for values, so setting 
            # the sample rate to 1 means the output is always replaced with an
            # attribute. So we're still getting attributes even though
            # the method is being fed content. 
            attributes =  get_minibatch(
                in_dataset['content'], out_dataset['tokenizer'], idx, 
                batch_size, max_len, idx=inputs[-1],
                dist_measurer=out_dataset['dist_measurer'], sample_rate=1.0)
        else:
            attributes =  get_minibatch(
                out_dataset['attribute'], out_dataset['tokenizer'], idx, 
                batch_size, max_len, idx=inputs[-1],
                dist_measurer=out_dataset['dist_measurer'], sample_rate=0.1)

    elif model_type == 'seq2seq':
        # ignore the in/out dataset stuff
        inputs = get_minibatch(
            src['data'], src['tokenizer'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            tgt['data'], tgt['tokenizer'], idx, batch_size, max_len, idx=inputs[-1])
        attributes = (None, None, None, None, None)

    else:
        raise Exception('Unsupported model_type: %s' % model_type)

    return inputs, attributes, outputs


def unsort(arr, idx):
    """unsort a list given idx: a list of each element's 'origin' index pre-sorting
    """
    unsorted_arr = arr[:]
    for i, origin in enumerate(idx):
        unsorted_arr[origin] = arr[i]
    return unsorted_arr



