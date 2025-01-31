import math
import numpy as np
import sys
from collections import Counter
from typing import List
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import editdistance
import heapq
from src import data
from src.cuda import CUDA
from src.callbacks import mean_masked_entropy
import random
import time
from modules.expectedMultiBleu import bleu as expected_bleu
from itertools import permutations
import pickle

import os
import logging
from utils.log_func import get_log_func
from flask import Flask
app = Flask(__name__)
app.logger.debug('debug')

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BLEU functions from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
#    (ran some comparisons, and it matches moses's multi-bleu.perl)
def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_edit_distance(hypotheses, reference):
    ed = 0
    for hyp, ref in zip(hypotheses, reference):
        ed += editdistance.eval(hyp, ref)

    return ed * 1.0 / len(hypotheses)

def update_training_index(cur_index, n_styles, batch_size):
    """ returns the next idx and sample size given current index, number of styles supported in 
        index region and batch size
    """
    train_sample_size = batch_size // n_styles
    cur_index -= train_sample_size
    if cur_index < 0:
        train_sample_size += cur_index
        cur_index = 0
    return cur_index, train_sample_size

def backpropagation_step(loss, optimizer, scheduler, scheduler_name, batch_idx, 
        update_frequency = 1, retain_graph = False):
    """ perform one step of backpropagation (supports accumulated gradients)"""
    loss.backward(retain_graph = retain_graph)

    if (batch_idx + 1) % update_frequency == 0:
        # every update_frequency batches update accumulated gradients
        optimizer.step()
        optimizer.zero_grad()
        if scheduler_name == 'cyclic' or scheduler_name == 'cosine':
            scheduler.step()

def define_optimizer_and_scheduler(lr, optimizer_type, scheduler_type, model, weight_decay = 0):
    """ define optimmizer and scheduler according to learning rate"""

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Learning method not recommended for this task")

    # reduce learning rate by a factor of 10 after plateau of 10 epochs
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif scheduler_type == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, 
            base_lr = lr / 10,  
            max_lr = lr,
            cycle_momentum = False,
            #step_size_up = 4000,
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max = 10
        )
    else:
        raise NotImplementedError("Learning scheduler not recommended for this task")
    return optimizer, scheduler

def generate_soft_sequence(max_len, start_id, model, content_data, attr_data, temperature, vocab_size):
    """ generate sequnce of prob. distributions over tokens to allow backprop through adversarial"""

    src_input, _, srclens, srcmask, _ = content_data
    aux_input, _, auxlens, auxmask, _ = attr_data

    # Initialize target with start_id for every sentence
    # start id must be dummy prob_dist over vocab_size
    start_token = torch.zeros(src_input.size(0), 1, vocab_size, dtype = torch.float)
    start_token[:, :, start_id] = 1
    tgt_input = start_token

    # initialize target mask for Transformer decoder
    tgt_mask = Variable(torch.BoolTensor(
        [
            [False] for i in range(src_input.size(0))
        ]
    ))

    if CUDA:
        start_token = start_token.cuda()
        tgt_input = tgt_input.cuda()
        tgt_mask = tgt_mask.cuda()
    for i in range(max_len):
        # run input through the model
        decoder_logit, _, decoder_states = model(src_input, tgt_input, 
            srcmask, srclens, aux_input, auxmask, auxlens, tgt_mask)
        # probability distribution is currently softmax(logits / temperature)
        tgt_input = torch.cat((start_token, model.softmax(decoder_logit / temperature)), dim=1)
        tgt_mask = srcmask[:, :i+2] # not sure how long the tgt_mask will be, so just copy srcmask
 
    return decoder_states

def calculate_discriminator_loss(dataset, style_ids, n_styles, content_data, attr_data, idx, tokenizer, model,
                                s_discriminators, config, decoder_states, sample_size, max_length):
    """ calculate discriminator loss over encoder states and tf decoder states vs. soft decoder states"""

    t = time.time()
    # sample minibatch from bt paradigm
    new_content, new_attr, _, out_dataset_ordering = data.minibatch(dataset, style_ids, n_styles, idx, sample_size, max_length, 
        config['model']['model_type'], is_adv = True)

    # generate sequences to compare to teacher-forced outputs from above
    input_lines_src, _, srclens, srcmask, _ = content_data
    input_ids_aux, _, auxlens, auxmask, _ = attr_data
    generated_decoder_states = generate_soft_sequence(
        max_length,
        tokenizer.bos_token_id, 
        model, 
        content_data, # this could also be new_content (they are the same)
        new_attr,
        config['model']['temperature'],
        len(tokenizer)
    )
    t1 = time.time()
    log(f'generating decoder states to different styles took: {t1 - t} seconds', level='debug')

    # shuffle decoder states according to sampled minibatch ordering
    shuffled_order = [i for j in out_dataset_ordering for i in range(j * sample_size, (j+1) * sample_size)]
    decoder_states_shuffled = decoder_states[shuffled_order]
    assert torch.all(torch.eq(decoder_states_shuffled[0], decoder_states[shuffled_order[0]]))

    # pass decoder states to discriminator module
    s_outputs = []
    for i, style in enumerate(style_ids):
        decoder_states_sample = decoder_states_shuffled[i * sample_size:(i+1) * sample_size]
        gen_decoder_states_sample = generated_decoder_states[i * sample_size:(i+1) * sample_size]
        s_outputs.append(s_discriminators[style].forward(torch.cat((decoder_states_sample, gen_decoder_states_sample), dim=0)))
    t2 = time.time()
    log(f'forward pass through discriminators took: {t2 - t1} seconds', level='debug')

    # calculate cross entropy loss over discriminators
    loss_criterion_d = nn.CrossEntropyLoss()
    # tf decoder states get label 1, soft decoder states get label 0
    decoder_labels = torch.cat((torch.ones(sample_size, dtype=torch.long), torch.zeros(sample_size, dtype = torch.long)))
    if CUDA:
        loss_criterion_d = loss_criterion_d.cuda()
        decoder_labels = decoder_labels.cuda()
    s_losses = [loss_criterion_d(style_output, decoder_labels) for style_output in s_outputs]
    t3 = time.time()
    log(f'calculating loss on discriminators took: {t3 - t2} seconds', level='debug')

    return s_losses

def calculate_loss(dataset, style_ids, n_styles, config, batch_idx, sample_size, max_length, model_type, model, 
                    s_discriminators, loss_crit = 'cross_entropy', bt_ratio = 1, is_test = False):
    """ sample minibatch, pass minibatch through model, calculate loss and entropy according to config"""

    src_packed, auxs_packed, tgt_packed = data.minibatch(dataset, style_ids, n_styles, batch_idx, 
        sample_size, max_length, model_type, is_test = is_test)
    input_lines_src, _, srclens, srcmask, _ = src_packed
    input_ids_aux, _, auxlens, auxmask, _ = auxs_packed
    input_lines_tgt, output_lines_tgt, tgtlens, tgtmask, _ = tgt_packed

    decoder_logit, decoder_probs, decoder_states = model(
        input_lines_src, input_lines_tgt, srcmask, srclens,
        input_ids_aux, auxlens, auxmask, tgtmask)
    
    # calculate loss on two minibatches separately, weight losses w/ ratio
    tokenizer = dataset[0]['tokenizer']
    weight_mask = torch.ones(len(tokenizer))
    if CUDA:
        weight_mask = weight_mask.cuda()
    weight_mask[tokenizer.pad_token_id] = 0
    
    # define loss criterion
    if loss_crit == 'cross_entropy':
        loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
        if CUDA:
            loss_criterion = loss_criterion.cuda()
    elif loss_crit != 'expected_bleu':
        raise NotImplementedError("Loss criterion not supported for this task")

    # calculate loss 
    if loss_crit == 'cross_entropy':
        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, len(tokenizer)),
            output_lines_tgt.view(-1)
        )
    else:
        # calculate lb expected bleu loss with max_order ngrams of 4 independent of ngram_range in config
        # decoder_logit.size()[1] is the max_length of a sequence for a given batch of translations
        loss = expected_bleu(decoder_probs, output_lines_tgt.cpu(), 
            torch.LongTensor([decoder_logit.size()[1]] * sample_size * len(style_ids)),
            tgtlens, smooth=True)[0]

    # mean entropy
    #mean_entropy = mean_masked_entropy(decoder_probs.data.cpu().numpy(), weight_mask.data.cpu().numpy, padding_id)

    # calculate discriminator loss if doing adversarial training, 
    if s_discriminators is not None:
        batch_len = len(src_packed[0]) // len(style_ids)
        s_losses = calculate_discriminator_loss(dataset, style_ids, n_styles, src_packed, auxs_packed, 
                batch_idx, tokenizer, model, s_discriminators, config, decoder_states, batch_len, max_length)
        loss = loss - config['training']['discriminator_ratio'] * sum(s_losses)
    else: 
        s_losses = None

    # get backtranslation minibatch (BT should be turned off for evaluation)
    if bt_ratio > 0 and not is_test:

        src_packed, auxs_packed, tgt_packed = data.back_translation_minibatch(dataset, style_ids, 
            n_styles, config, batch_idx, sample_size, max_length, model, model_type)
        bt_input_lines_src, _, bt_srclens, bt_srcmask, _ = src_packed
        bt_input_ids_aux, _, bt_auxlens, bt_auxmask, _ = auxs_packed
        bt_input_lines_tgt, bt_output_lines_tgt, bt_tgtlens, bt_tgtmask, _ = tgt_packed

        bt_decoder_logit, bt_decoder_probs, bt_decoder_states = model(
            bt_input_lines_src, bt_input_lines_tgt, bt_srcmask, bt_srclens,
            bt_input_ids_aux, bt_auxlens, bt_auxmask, bt_tgtmask)
        
        # calculate loss
        if loss_crit == 'cross_entropy':
            bt_loss = loss_criterion(
                bt_decoder_logit.contiguous().view(-1, len(tokenizer)),
                bt_output_lines_tgt.view(-1)
            )
        else:
            bt_loss = expected_bleu(bt_decoder_probs, bt_output_lines_tgt.cpu(), 
                torch.LongTensor([bt_decoder_logit.size()[1]] * sample_size * len(style_ids)),
                bt_tgtlens, smooth=True)[0]

        # calculate discriminator loss if doing adversarial training
        if s_discriminators is not None:
            bt_s_losses = calculate_discriminator_loss(dataset, style_ids, n_styles, src_packed, auxs_packed, 
                    batch_idx, tokenizer, model, s_discriminators, config, bt_decoder_states, sample_size, max_length)
            s_losses = [(bt_ratio * bt_s_loss + s_loss) / 2 for bt_s_loss, s_loss in zip(bt_s_losses, s_losses)]
            bt_loss = bt_loss - config['training']['discriminator_ratio'] * sum(s_losses)

        # combine losses
        loss = (bt_ratio * bt_loss + loss) / 2

        # mean entropy
        #bt_mean_entropy = mean_masked_entropy(bt_decoder_probs.data.cpu().numpy(), weight_mask.data.cpu().numpy, padding_id)
        #mean_entropy = (bt_ratio * bt_mean_entropy + mean_entropy) / 2
 
    # return combined loss, discrim loss, and combined mean entropy  
    return loss, s_losses#, mean_entropy

def decode_minibatch_greedy(max_len, start_id, stop_id, model, src_input, srclens, srcmask,
        aux_input, auxlens, auxmask):
    """ argmax decoding """

    # Initialize target with start_id for every sentence
    tgt_input = Variable(torch.LongTensor(
        [
            [start_id] for i in range(src_input.size(0))
        ]
    ))

    # initialize target mask for Transformer decoder
    tgt_mask = Variable(torch.BoolTensor(
        [
            [False] for i in range(src_input.size(0))
        ]
    ))

    if CUDA:
        tgt_input = tgt_input.cuda()
        tgt_mask = tgt_mask.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logit, word_probs, decoder_states = model(src_input, tgt_input, 
            srcmask, srclens, aux_input, auxlens, auxmask, tgt_mask)
        decoder_argmax = word_probs.data.cpu().numpy()[:,-1,:].argmax(axis=-1)
        
        # select the predicted "next" tokens, attach to target-side inputs
        next_preds = Variable(torch.from_numpy(decoder_argmax))
        prev_mask = tgt_mask.data.cpu().numpy()[:,-1]
        next_mask = [[True] if cur == [stop_id] or prev == [True] else [False] for cur, prev in zip(decoder_argmax, prev_mask)]
        next_mask_unrolled = [val for val_list in next_mask for val in val_list]
        if CUDA:
            next_preds = next_preds.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

        # check for early stopping to speed up decoding
        if sum(next_mask_unrolled) == len(next_mask_unrolled):
            return tgt_input
        else:
            next_mask = torch.BoolTensor(next_mask)
            if CUDA:
                next_mask = next_mask.cuda()
            tgt_mask = torch.cat((tgt_mask, next_mask), dim=1)
    return tgt_input

def ids_to_toks(tok_seqs, tokenizer, sort = True, indices = None):
    """ convert seqs to tokens"""

    # take off the gpu
    tok_seqs = tok_seqs.cpu().numpy()

    # convert to toks, delete any special tokens (bos, eos, pad)
    tok_seqs = [line[1:] if line[0] == tokenizer.bos_token_id else line for line in tok_seqs]
    tok_seqs = [np.split(line, np.where(line == tokenizer.eos_token_id)[0])[0] for line in tok_seqs]
    tok_seqs = [tokenizer.decode(line) for line in tok_seqs]

    # unsort
    if sort:
        return data.unsort(tok_seqs, indices)
    else:
        return tok_seqs

def generate_sequences(tokenizer, model, config, start_id, stop_id, input_content, input_aux):
    "generate sequences of output by sampling from token distribution according to decoding strategy"
    
    input_lines_src, _, srclens, srcmask, _ = input_content
    input_ids_aux, _, auxlens, auxmask, _ = input_aux

    # decode dataset with greedy, beam search, or top k
    start_time = time.time()
    if config['model']['decode'] == 'greedy':
        tgt_pred = decode_minibatch_greedy(
            config['data']['max_len'], start_id, stop_id, 
            model, input_lines_src, srclens, srcmask,
            input_ids_aux, auxlens, auxmask)
        log(f'greedy search decoding took: {time.time() - start_time}', level='debug')
    # elif config['model']['decode'] == 'beam_search':
    #     start_time = time.time()
    #     if config['model']['model_type'] == 'delete_retrieve':
    #         tgt_pred = torch.stack([beam_search_decode(
    #             config['data']['max_len'], start_id, stop_id, model, 
    #             i, [i_l], i_m, a, [a_l], a_m,
    #             data.get_padding_id(tokenizer), config['model']['beam_width']) for 
    #             i, i_l, i_m, a, a_l, a_m in zip(input_lines_src, srclens, srcmask,
    #             input_ids_aux, auxlens, auxmask)])
    #     elif config['model']['model_type'] == 'delete':
    #         input_ids_aux = input_ids_aux.unsqueeze(1)
    #         tgt_pred = torch.stack([beam_search_decode(
    #             config['data']['max_len'], start_id, stop_id, model, 
    #             i, [i_l], i_m, a, None, None,
    #             data.get_padding_id(tokenizer), config['model']['beam_width']) for 
    #             i, i_l, i_m, a in zip(input_lines_src, srclens, srcmask,
    #             input_ids_aux)])
    #     log(f'beam search decoding took: {time.time() - start_time}', level='debug')
    elif config['model']['decode'] == 'top_k':
        start_time = time.time()
        tgt_pred = decode_top_k(
            config['data']['max_len'], start_id, stop_id,
            model, input_lines_src, srclens, srcmask,
            input_ids_aux, auxlens, auxmask, 
            config['model']['k'], config['model']['temperature']
        )
        log(f'top k decoding took: {time.time() - start_time}', level='debug')

    else:
        raise Exception('Decoding method must be one of greedy or top_k')

    return tgt_pred

def decode_dataset(model, test_data, sample_size, num_samples, config):
    """Evaluate model on num_samples of size sample_size"""

    content_lengths = [len(datum['content']) for datum in test_data]
    style_ids = [i for i in range(len(content_lengths))]
    min_content_length = min(content_lengths)
    upper_lim = min(min_content_length, num_samples * sample_size)

    # create list of lists to separate translations for each style
    inputs = [[] for i in range(len(style_ids))]
    preds = [[] for i in range(len(style_ids))]
    auxs = [[] for i in range(len(style_ids))]
    ground_truths = [[] for i in range(len(style_ids))]

    for j in range(0, upper_lim, sample_size):
        sys.stdout.write("\r%s/%s..." % (j * len(style_ids), upper_lim * len(style_ids)))
        sys.stdout.flush()

        # get batch
        if j + sample_size > min_content_length:
            sample_size = min_content_length - j
        src_packed, auxs_packed, tgt_packed = data.minibatch(test_data, style_ids, len(style_ids), j, 
            sample_size, config['data']['max_len'], config['model']['model_type'], is_test = True)
        _, output_lines_src, _, _, indices = src_packed
        input_ids_aux, _, _, _, _ = auxs_packed
        _, output_lines_tgt, _, _, _ = tgt_packed
        
        # generate sequences according to decoding strategy
        tokenizer = test_data[0]['tokenizer']
        tgt_pred = generate_sequences(
            tokenizer,
            model, 
            config,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            src_packed,
            auxs_packed,
        )
        # convert inputs/preds/targets/aux to human-readable form
        for i in range(len(test_data)):
            inputs[i] += ids_to_toks(output_lines_src[i*sample_size:(i+1)*sample_size], tokenizer, sort = False)
            preds[i] += ids_to_toks(tgt_pred[i*sample_size:(i+1)*sample_size], tokenizer, sort = False)
            ground_truths[i] += ids_to_toks(output_lines_tgt[i*sample_size:(i+1)*sample_size], tokenizer, sort = False)
    
            if config['model']['model_type'] == 'delete':
                auxs[i] += [[str(x[0])] for x in input_ids_aux[i*sample_size:(i+1)*sample_size].data.cpu().numpy()] # because of list comp in inference_metrics()
            elif config['model']['model_type'] == 'delete_retrieve':
                auxs[i] += ids_to_toks(input_ids_aux[i*sample_size:(i+1)*sample_size].squeeze(-1), tokenizer, sort = False)
            elif config['model']['model_type'] == 'seq2seq':
                auxs[i] += ['None' for _ in range(sample_size)]
    return inputs, preds, ground_truths, auxs


def inference_metrics(model, s_discriminators, test_data, sample_size, num_samples, config):
    """ decode and evaluate bleu """
    inputs, preds, ground_truths, auxs = decode_dataset(
        model, test_data, sample_size, num_samples, config)
    bleus = [get_bleu(p, truth) for p, truth in zip(preds, ground_truths)]
    edit_distances = [get_edit_distance(p, truth) for p, truth in zip(preds, ground_truths)]

    inputs = [''.join(seq) for ins in inputs for seq in ins]
    preds = [''.join(seq) for p in preds for seq in p]
    ground_truths = [''.join(seq) for g in ground_truths for seq in g]
    auxs = [''.join(seq) for a in auxs for seq in a]

    # put models back in train mode
    model.train()
    if s_discriminators is not None:
        [discriminator.train() for discriminator in s_discriminators]

    return bleus, edit_distances, inputs, preds, ground_truths, auxs

def evaluate_lpp(model, s_discriminators, test_data, sample_size, config):
    """ evaluate log perplexity WITHOUT decoding
        (i.e., with teacher forcing)
    """

    # put models in eval mode
    model.eval()
    if s_discriminators is not None:
        [discriminator.eval() for discriminator in s_discriminators]

    losses = []
    d_losses = [[]]
    content_lengths = [len(datum['content']) for datum in test_data]
    style_ids = [i for i in range(len(content_lengths))]
    
    # round down from sample size
    min_content_length = min(content_lengths)
    for j in range(0, min_content_length, sample_size):
        sys.stdout.write("\r%s/%s..." % (j, min_content_length))
        sys.stdout.flush()

        loss_crit = config['training']['loss_criterion']
        if j + sample_size > min_content_length:
            sample_size = min_content_length - j
        combined_loss, s_losses = calculate_loss(test_data, style_ids, len(content_lengths), config, j, sample_size, config['data']['max_len'], 
            config['model']['model_type'], model, s_discriminators, loss_crit=loss_crit, bt_ratio=config['training']['bt_ratio'], is_test=True)

        loss_item = combined_loss.item() if loss_crit == 'cross_entropy' else -combined_loss.item()
        losses.append(loss_item)

        if s_losses is not None: 
            [d_loss.append(s_loss.item()) for d_loss, s_loss in zip(d_losses, s_losses)]
            d_means = [np.mean(d_loss) for d_loss in d_losses]
        else:
            d_means = None
    return np.mean(losses), d_means

def predict_text(input_content, tokenizer, style_ids, config, model, k = 5, temperature = 1.0, 
        number_preds = 1, train_data = None):
    """ translate input sequence (not in train / test corpora) to another style (s). 
        train_data only necessary in delete and retrieve framework to create tfidf similarity
        between input_text and training corpii to extract appropriate attributes for embedding. 
    """

    start_time = time.time()

    # remove attribute tokens from input according if they have been marked
    # (if word / ngram attributes pre-calculated they need to be cached and copied to image)
    input_content = input_content.split()
    if config['data']['noise'] == 'word_attributes' or config['data']['noise'] == 'ngram_attributes':
        attr_path = os.path.join("checkpoints", config['data']['vocab_dir'], 'style_vocabs.pkl')
        attrs = pickle.load(open(attr_path, "rb"))

        # remove attributes from all styles
        all_attrs = [attr for attr_list in attrs for attr in attr_list]
        # permutation and dropout_prob should also be turned off for deployment inference
        _, input_content, _ = data.extract_attributes(input_content, all_attrs, config['data']['noise'], 
            0, config['data']['ngram_range'], 0)
    
    if config['model']['model_type'] == 'delete_retrieve':
        # get attribute examples by measuring distance to train_data corpii with tfidf
        train_data = [train_data[style_id] for style_id in style_ids]
        dist_measurers = [data.CorpusSearcher(
            query_corpus=[' '.join(x) for x in input_content],
            key_corpus=[' '.join(x) for x in train_dict['content']],
            value_corpus=[' '.join(x) for x in train_dict['attribute']],
            vectorizer=data.TfidfVectorizer(),
            make_binary=False)
            for train_dict in train_data]

        # last two args are sample rate (always 1.0) and key_idx (always 0)
        input_attributes = [data.sample_replace([input_content], tokenizer, [dist_m], 1.0, 0) for dist_m in dist_measurers]

        # tokenize attributes
        input_attr = torch.LongTensor([
            [tokenizer.bos_token_id] + 
            tokenizer.encode(" ".join(attr))[:config['data']['max_len'] - 2]
            for attr in input_attributes])
        attr_length = [max([attr.shape[1] for attr in input_attr])]
        attr_mask = torch.BoolTensor([([False] * attr_length)])

    else:
        input_attributes = torch.LongTensor([style_ids])
        attr_length = None
        attr_mask = None

    # tokenize content 
    input_content = torch.LongTensor([
        [tokenizer.bos_token_id] + 
        tokenizer.encode(" ".join(input_content))[:config['data']['max_len'] - 2]])
    content_length = [input_content.shape[1]]
    content_mask = torch.BoolTensor([([False] * content_length[0])])

    # copy # of predictions times for batch decoding
    input_attributes = input_attributes.repeat(number_preds, 1)
    input_content = input_content.repeat(number_preds, 1)
    content_mask = content_mask.repeat(number_preds, 1)
    content_length = content_length * number_preds
    if attr_length is not None:
        attr_mask = attr_mask.repeat(number_preds, 1)
        attr_length = attr_length * number_preds

    # decode according to decoding strategy
    content = (input_content, None, content_length, content_mask, None)
    attributes = (input_attributes, None, attr_length, attr_mask, None)
    t0 = time.time()

    # update k and temperature with user inputs, generate sequence
    config['model']['k'] = k
    config['model']['temperature'] = temperature
    output = generate_sequences(tokenizer, model, config, tokenizer.bos_token_id, tokenizer.eos_token_id, content, attributes)

    # convert tokens to text
    preds = ids_to_toks(output, tokenizer, sort=False)
    log(preds, level='debug')
    return preds

def sample_softmax(logits, temperature=1.0, num_samples=1):
    """ sample from softmax distribution over tokens with temperature"""

    exps = np.exp((logits - np.max(logits)) / temperature)
    probs = exps / np.sum(exps)
    return np.random.choice(logits.shape[0], p = probs)

def get_next_token_scores(model, src_input, tgt_input, srcmask, srclen, 
                        aux_input, auxmask, auxlen):
    """ get next token logit probabilities"""

    # get tensors in correct shape for prediction
    src_input = src_input.unsqueeze(0)
    srcmask = srcmask.unsqueeze(0)
    if auxmask is not None:
        auxmask = auxmask.unsqueeze(0)
    if CUDA:
        tgt_input = tgt_input.cuda()
    decoder_logit, word_probs, _ = model(src_input, tgt_input, srcmask, srclen,
        aux_input, auxmask, auxlen)
    return decoder_logit.data.cpu().numpy()[0, -1, :]

def decode_top_k(
    max_len,
    start_id,
    stop_id,
    model,
    src_input,
    srclens,
    srcmask,
    aux_input,
    auxlens,
    auxmask,
    k = 10,
    temperature=1.0
):
    """ perform top k decoding according to k and temperature params"""

    # Initialize target with start_id for every sentence
    tgt_input = Variable(torch.LongTensor(
        [
            [start_id] for i in range(src_input.size(0))
        ]
    ))

    # initialize target mask for Transformer decoder
    tgt_mask = Variable(torch.BoolTensor(
        [
            [False] for i in range(src_input.size(0))
        ]
    ))

    if CUDA:
        tgt_input = tgt_input.cuda()
        tgt_mask = tgt_mask.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logits, _, decoder_states = model(src_input, tgt_input, srcmask, srclens,
            aux_input, auxlens, auxmask, tgt_mask)
        decoder_logits = decoder_logits.data.cpu().numpy()[:,-1,:]

        # if k=1, do greedy sampling
        if k == 1:
            sampled_indices = decoder_logits.argmax(axis=-1)
        # if k > 1, do softmax sampling over the top-k
        elif k:
            # grab last k (largest scores)
            top_ids = decoder_logits.argsort(axis = -1)[:,-k:]
            top_scores = [x[idx] for x, idx in zip(decoder_logits, top_ids)]
            inds = [sample_softmax(top, temperature=temperature) for top in top_scores]
            sampled_indices = np.array([x[idx] for x,idx in zip(top_ids, inds)])
        next_preds = torch.LongTensor(sampled_indices)
        prev_mask = tgt_mask.data.cpu().numpy()[:,-1]
        next_mask = [[True] if cur == [stop_id] or prev == [True] else [False] for cur, prev in zip(sampled_indices, prev_mask)]
        next_mask_unrolled = [val for val_list in next_mask for val in val_list]
        if CUDA:
            next_preds = next_preds.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

        # check for early stopping to speed up decoding
        if sum(next_mask_unrolled) == len(next_mask_unrolled):
            return tgt_input
        else:
            next_mask = torch.BoolTensor(next_mask)
            if CUDA:
                next_mask = next_mask.cuda()
            tgt_mask = torch.cat((tgt_mask, next_mask), dim=1)
    return tgt_input

    
    
class Beam(object):
    """ Beam object for beam search decoding"""

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, score, complete, prefix):
        heapq.heappush(self.heap, (score, complete, prefix))
        # except RuntimeError:
        #     print(f'Heap push failed')
        #     print(f'score: {score}')
        #     print(f'complete: {complete}')
        #     print(f'prefix size: {prefix.size()}')
        #     print(f'prefix: {prefix}')
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)
       
    def __iter__(self):
        return iter(self.heap)

def beam_search_decode(
    max_len,
    start_id,
    stop_id,
    model,
    src_input,
    srclens,
    srcmask,
    aux_input,
    auxlens,
    auxmask,
    padding_id,
    beam_width=10,
    num_return=1,
    return_scores=False
):
    """ Beam search decoding method 
    
    Arguments:
        src_input {List(int)} -- [Input sequence of integers]
    
    Keyword Arguments:
        beam_width {int} -- [Number of beams/sequences to use in search (higher can get better responses, but takes longer)] (default: {10})
    
    Returns:
        [(float, list[int])] -- sorted list of (score, sequence) pairs
        
    """
    s = time.time()
    prev_beam = Beam(beam_width)
    prev_beam.add(1.0, False, Variable(torch.LongTensor([[start_id]])))

    while True:
        curr_beam = Beam(beam_width)
        # Add complete sentences to the current beam, add more words to the rest
        for (prefix_score, complete, prefix) in prev_beam:
            if complete:
                curr_beam.add(prefix_score, True, prefix)
            else:
                # run input through the model
                decoder_logits = get_next_token_scores(model, src_input, prefix, srcmask, srclens,
                    aux_input, auxlens, auxmask)
                for next_id, next_score in enumerate(decoder_logits):
                    score = prefix_score + next_score
                    next_pred = Variable(torch.from_numpy(np.array([[next_id]])))
                    new_prefix = torch.cat((prefix, next_pred), dim=1)
                    # could pad after stacking, beam search to slow currently anyways...
                    #now_complete = next_id == stop_id or new_prefix.size()[1] >= max_len
                    now_complete = new_prefix.size()[1] >= max_len
                    curr_beam.add(score, now_complete, new_prefix)
        
            # if all beams are completed, sort and return (score, seq) pairs
            if all([complete for _, complete, _ in curr_beam]):
                curr_beam = sorted(curr_beam, reverse=True)[:num_return]

                # pad beams that completed early 
                #lens = [prefix.size()[1] for _, _, prefix in curr_beam] 
                #max_len = max(lens)
                #padding = [Variable(torch.from_numpy(np.array([[padding_id] * (max_len - l)]))) for l in lens]
                #padded = [(score, torch.cat((prefix, pad), dim=1)) for (score, _, prefix), pad in zip(curr_beam, padding)]
                
                if return_scores:
                    generated_seqs = [(score, prefix) for score, _, prefix in curr_beam]
                else:
                    generated_seqs = [prefix for score, _, prefix in curr_beam]
                if num_return == 1:
                    return generated_seqs[0]
                else:
                    return generated_seqs

            prev_beam = curr_beam

    
