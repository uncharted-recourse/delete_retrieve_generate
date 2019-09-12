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

import os
import logging
from utils.log_func import get_log_func

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

def backpropagation_step(loss, optimizer, retain_graph = False):
    """ perform one step of backpropagation"""
    optimizer.zero_grad()
    loss.backward(retain_graph = retain_graph)
    optimizer.step()

def define_optimizer_and_scheduler(lr, optimizer_type, scheduler_type, model):
    """ define optimmizer and scheduler according to learning rate"""

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Learning method not recommended for this task")

    # reduce learning rate by a factor of 10 after plateau of 10 epochs
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif scheduler_type == 'cyclic':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            base_lr = lr,  
            max_lr = 10 * lr
        )
    else:
        raise NotImplementedError("Learning scheduler not recommended for this task")
    return optimizer, scheduler

def calculate_discriminator_loss(dataset, content_data, attr_data, idx, tokenizer, model, z_discriminator,
                                s_discriminators, config, decoder_states, sample_size, max_length):
    """ calculate discriminator loss over encoder states and tf decoder states vs. soft decoder states"""

    t = time.time()
    # sample minibatch from bt paradigm (next style for now)
    new_content, new_attr, _, out_dataset_ordering = minibatch(dataset, idx, sample_size, max_length, 
        config['model']['model_type'], is_bt = True)

    # generate sequences to compare to teacher-forced outputs from above
    _, generated_decoder_states = generate_sequences(
        tokenizer,
        model, 
        config,
        data.get_start_id(tokenizer),
        data.get_stop_id(tokenizer),
        content_data, # this could also be new_content (they are the same)
        new_attr,
    )
    t1 = time.time()
    log(f'generating decoder states to different styles took: {t1 - t} seconds', level='info')

    # shuffle decoder states according to sampled minibatch ordering
    shuffled_order = [i for j in out_dataset_ordering for i in range(j * sample_size, (j+1) * sample_size)]
    decoder_states_shuffled = decoder_states[shuffled_order]
    assert decoder_states_shuffled[0] == decoder_states[shuffled_order[0]]

    # pass decoder states to discriminator module
    s_outputs = []
    for i in range(len(s_discriminators)):
        decoder_states_sample = decoder_states_shuffled[i * sample_size:(i+1) * sample_size]
        gen_decoder_states_sample = generated_decoder_states[i * sample_size:(i+1) * sample_size]
        s_outputs.append(s_discriminators[i].forward(torch.cat((decoder_states_sample, gen_decoder_states_sample), dim=0)))
    t2 = time.time()
    log(f'forward pass through discriminators took: {t2 - t1} seconds', level='info')

    # calculate cross entropy loss over discriminators
    loss_criterion_d = nn.CrossEntropyLoss()
    # tf decoder states get label 1, soft decoder states get label 0
    decoder_labels = torch.cat((torch.ones(sample_size, dtype=torch.long), torch.zeros(sample_size, dtype = torch.long)))
    if CUDA:
        loss_criterion_d = loss_criterion_d.cuda()
        decoder_labels = decoder_labels.cuda()
    s_losses = [loss_criterion_d(style_output, decoder_labels) for style_output in s_outputs]
    t3 = time.time()
    log(f'calculating loss on discriminators took: {t3 - t2} seconds', level='info')

    return s_losses

def calculate_loss(dataset, n_styles, config, batch_idx, sample_size, max_length, model_type, model, 
                    s_discriminators, loss_crit = 'cross_entropy', bt_ratio = 1, is_test = False):
    """ sample minibatch, pass minibatch through model, calculate loss and entropy according to config"""

    # sample even number of samples from each corpus according to batch size, 
    src_packed, auxs_packed, tgt_packed = data.minibatch(dataset, batch_idx, 
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
    padding_id = data.get_padding_id(tokenizer)
    weight_mask[padding_id] = 0
    
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
            torch.LongTensor([decoder_logit.size()[1]] * sample_size * n_styles),
            tgtlens, smooth=True)[0]

    # mean entropy
    mean_entropy = mean_masked_entropy(decoder_probs.data.cpu().numpy(), weight_mask.data.cpu().numpy, padding_id)

    # calculate discriminator loss if doing adversarial training, 
    if s_discriminators is not None:
        s_losses = calculate_discriminator_loss(dataset, src_packed, auxs_packed, batch_idx, tokenizer, 
                model, s_discriminators, config, decoder_states, sample_size, max_length)
        loss = loss - config['training']['discriminator_ratio'] * sum(s_losses)
    else: 
        s_losses = None

    # get backtranslation minibatch (BT should be turned off for evaluation)
    if bt_ratio > 0 and not is_test:

        src_packed, auxs_packed, tgt_packed = data.back_translation_minibatch(dataset, batch_idx, 
            sample_size, max_length, model_type, is_bt = True)
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
                torch.LongTensor([bt_decoder_logit.size()[1]] * sample_size * n_styles),
                bt_tgtlens, smooth=True)[0]

        # calculate discriminator loss if doing adversarial training
        if z_discriminator is not None:
            bt_s_losses = calculate_discriminator_loss(dataset, src_packed, auxs_packed, batch_idx, tokenizer, 
                    model, s_discriminators, config, bt_decoder_states, sample_size, max_length)
            s_losses = [(bt_ratio * bt_s_loss + s_loss) / 2 for bt_s_loss, s_loss in zip(bt_s_losses, s_losses)]
            bt_loss = bt_loss - config['training']['discriminator_ratio'] * sum(s_losses)

        # combine losses
        loss = (bt_ratio * bt_loss + loss) / 2

        # mean entropy
        bt_mean_entropy = mean_masked_entropy(bt_decoder_probs.data.cpu().numpy(), weight_mask.data.cpu().numpy, padding_id)
        mean_entropy = (bt_ratio * bt_mean_entropy + mean_entropy) / 2
 
    # return combined loss, discrim loss, and combined mean entropy  
    return loss, s_losses, mean_entropy

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
    tgt_mask = Variable(torch.LongTensor(
        [
            [1] for i in range(src_input.size(0))
        ]
    ))

    if CUDA:
        tgt_input = tgt_input.cuda()
        tgt_mask = tgt_mask.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logit, word_probs, decoder_states = model(src_input, tgt_input, 
            srcmask, srclens, aux_input, auxmask, auxlens, tgt_mask)
        decoder_argmax = word_probs.data.cpu().numpy()[:,-1,:].argmax(axis=-1)
        
        # select the predicted "next" tokens, attach to target-side inputs
        next_preds = Variable(torch.from_numpy(decoder_argmax))
        prev_mask = tgt_mask.data.cpu().numpy()[:,-1]
        next_mask = [[0] if cur == [stop_id] or prev == [0] else [1] 
            for cur, prev in zip(decoder_argmax, prev_mask)]
        next_mask = Variable(torch.from_numpy(np.array(next_mask)))
        if CUDA:
            next_preds = next_preds.cuda()
            next_mask = next_mask.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)
        tgt_mask = torch.cat((tgt_mask, next_mask), dim=1)

    # return decoder_states (argmax sampling)
    return tgt_input, decoder_states

def ids_to_toks(tok_seqs, tokenizer, sort = True, indices = None):
    """ convert seqs to tokens"""

    # take off the gpu
    tok_seqs = tok_seqs.cpu().numpy()
    # convert to toks, delete any special tokens (bos, eos, pad)
    start_id = data.get_start_id(tokenizer)
    stop_id = data.get_stop_id(tokenizer)
    tok_seqs = [line[1:] if line[0] == start_id else line for line in tok_seqs]
    tok_seqs = [np.split(line, np.where(line == stop_id)[0])[0] for line in tok_seqs]
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
        tgt_pred, decoder_states = decode_minibatch_greedy(
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
        tgt_pred, decoder_states = decode_top_k(
            config['data']['max_len'], start_id, stop_id,
            model, input_lines_src, srclens, srcmask,
            input_ids_aux, auxlens, auxmask, 
            config['model']['k'], config['model']['temperature']
        )
        log(f'top k decoding took: {time.time() - start_time}', level='debug')
    else:
        raise Exception('Decoding method must be one of greedy or top_k')

    return tgt_pred, decoder_states

def decode_dataset(model, test_data, sample_size, num_samples, config):
    """Evaluate model on num_samples of size sample_size"""

    inputs = []
    preds = []
    auxs = []
    ground_truths = []

    content_lengths = [len(datum['content']) for datum in test_data]
    min_content_length = min(content_lengths)
    upper_lim = min(min_content_length, num_samples * sample_size)
    for j in range(0, upper_lim, sample_size):
        sys.stdout.write("\r%s/%s..." % (j, upper_lim))
        sys.stdout.flush()

        # get batch
        src_packed, auxs_packed, tgt_packed = data.minibatch(test_data, j, 
            sample_size, config['data']['max_len'], config['model']['model_type'], is_test = True)
        _, output_lines_src, _, _, indices = src_packed
        input_ids_aux, _, _, _, _ = auxs_packed
        _, output_lines_tgt, _, _, _ = tgt_packed
        
        # generate sequences according to decoding strategy
        tokenizer = test_data[0]['tokenizer']
        tgt_pred, _ = generate_sequences(
            tokenizer,
            model, 
            config,
            data.get_start_id(tokenizer),
            data.get_stop_id(tokenizer),
            src_packed,
            auxs_packed,
        )

        # convert inputs/preds/targets/aux to human-readable form
        inputs += ids_to_toks(output_lines_src, tokenizer, indices=indices)
        preds += ids_to_toks(tgt_pred, tokenizer, indices=indices)
        ground_truths += ids_to_toks(output_lines_tgt, tokenizer, indices=indices)
        
        if config['model']['model_type'] == 'delete':
            auxs += [[str(x)] for x in input_ids_aux.data.cpu().numpy()] # because of list comp in inference_metrics()
        elif config['model']['model_type'] == 'delete_retrieve':
            auxs += ids_to_toks(input_ids_aux, tokenizer, indices = indices)
        elif config['model']['model_type'] == 'seq2seq':
            auxs += ['None' for _ in range(len(tgt_pred))]

    return inputs, preds, ground_truths, auxs


def inference_metrics(model, test_data, sample_size, num_samples, config):
    """ decode and evaluate bleu """
    inputs, preds, ground_truths, auxs = decode_dataset(
        model, test_data, sample_size, num_samples, config)
    bleu = get_bleu(preds, ground_truths)
    edit_distance = get_edit_distance(preds, ground_truths)

    inputs = [''.join(seq) for seq in inputs]
    preds = [''.join(seq) for seq in preds]
    ground_truths = [''.join(seq) for seq in ground_truths]
    auxs = [''.join(seq) for seq in auxs]

    return bleu, edit_distance, inputs, preds, ground_truths, auxs


def evaluate_lpp(model, z_discriminator, s_discriminators, test_data, sample_size, config):
    """ evaluate log perplexity WITHOUT decoding
        (i.e., with teacher forcing)
    """
    losses = []
    d_losses = [[]]
    content_lengths = [len(datum['content']) for datum in test_data]
    min_content_length = min(content_lengths)
    for j in range(0, min_content_length, sample_size):
        sys.stdout.write("\r%s/%s..." % (j, min_content_length))
        sys.stdout.flush()

        loss_crit = config['training']['loss_criterion']
        combined_loss, s_losses, combined_mean_entropy = calculate_loss(test_data, len(content_lengths), config, j, sample_size, config['data']['max_len'], 
            config['model']['model_type'], model, z_discriminator, s_discriminators, loss_crit=loss_crit, bt_ratio=config['training']['bt_ratio'], is_test=True)

        loss_item = combined_loss.item() if loss_crit == 'cross_entropy' else -combined_loss.item()
        losses.append(loss_item)

        if s_losses is not None: 
            [d_loss.append(s_loss.item()) for d_loss, s_loss in zip(d_losses, s_los_losses)]
            d_means = [np.mean(d_loss) for d_loss in d_losses]
        else:
            d_means = None
    return np.mean(losses), d_means, combined_mean_entropy

def predict_text(text, model, src, tgt, config, cache_dir = None, forward = True, remove_attributes = True):
    
    start_time = time.time()

    # tokenize input data using cached tokenizer and attribute vocab
    tokenized_text, tokenizer = data.encode_text_data([text],
        encoder = config['data']['tokenizer'], 
        cache_dir=cache_dir
    )
    if forward:
        attr_path = os.path.join(cache_dir, 'pre_attribute_vocab.pkl')
    else:
        attr_path = os.path.join(cache_dir, 'post_attribute_vocab.pkl')
    attr = pickle.load(open(attr_path, "rb"))

    if remove_attributes:
        src_lines, src_content, src_attribute = list(zip(
            *[data.extract_attributes(line, attr, config['data']['noise'], config['data']['dropout_prob'],
                config['data']['ngram_range'], config['data']['permutation']) for line in tokenized_text]
        ))
    
    # convert content to tokens
    max_len = config['data']['max_len']
    start_id = data.get_start_id(tokenizer)
    stop_id = data.get_stop_id(tokenizer)
    lines = [[start_id] + l[:max_len] + [stop_id] for l in input_data]
    content_length = [len(l) - 1 for l in lines]
    content_mask = [([1] * l) for l in content_length]
    content = [[int(w) for w in l[:-1]] for l in lines]

    # convert attributes to tokens or binary variables
    attributes_len = None
    attributes_mask = None
    if config['model']['model_type'] == 'delete':
        if forward:
            attributes = [1]
        else:
            attributes = [0]
    elif config['model']['model_type'] == 'delete_retrieve':
        if forward:
            attributes = data.sample_replace(lines, tgt['dist_measurer'], 1.0, 0)
        else:
            attributes = data.sample_replace(lines, src['dist_measurer'], 1.0, 0)
        attributes = [[int(w) for w in l[:-1]] for l in attributes]
        attributes_len = [len(l) - 1 for l in attributes]
        attributes_mask = [([1] * l) for l in attributes_len]
        attributes_mask = Variable(torch.LongTensor(attributes_mask))
    else:
        raise Exception('Currently only "delete" and "delete_retrieve" models are supported for predict_text()')
    log(f'time for tokenization: {time.time() - start_time}', level='debug')
    
    # convert to torch objects for prediction
    content = Variable(torch.LongTensor(content))
    content_mask = Variable(torch.FloatTensor(content_mask))
    attributes = Variable(torch.LongTensor(attributes))

    if CUDA:
        content = content.cuda()
        content_mask = content_mask.cuda()
        attributes = attributes.cuda()
        attributes_mask = attributes_mask.cuda()

    # make predictions
    model.eval()
    if config['model']['decode'] == 'greedy':
        tgt_pred = decode_minibatch_greedy(
            max_len, start_id, 
            model, content, content_length, content_mask,
            attributes, attributes_len, attributes_mask
        )
    elif config['model']['decode'] == 'beam_search':
        start_time = time.time()
        if config['model']['model_type'] == 'delete_retrieve':
            tgt_pred = torch.stack([beam_search_decode(
                config['data']['max_len'], start_id, stop_id, model, 
                i, [i_l], i_m, a, [a_l], a_m,
                data.get_padding_id(tokenizer), config['model']['beam_width']) for 
                i, i_l, i_m, a, a_l, a_m in zip(input_lines_src, srclens, srcmask,
                input_ids_aux, auxlens, auxmask)])
        elif config['model']['model_type'] == 'delete':
            input_ids_aux = input_ids_aux.unsqueeze(1)
            tgt_pred = torch.stack([beam_search_decode(
                config['data']['max_len'], start_id, stop_id, model, 
                i, [i_l], i_m, a, None, None,
                data.get_padding_id(tokenizer), config['model']['beam_width']) for 
                i, i_l, i_m, a in zip(input_lines_src, srclens, srcmask,
                input_ids_aux)])
        log(f'beam search decoding took: {time.time() - start_time}', level='debug')
    elif config['model']['decode'] == 'top_k':
        start_time = time.time()
        tgt_pred = decode_top_k(
            max_len, start_id, stop_id,
            model, content, content_length, content_mask,
            attributes, attributes_len, attributes_mask, 
            config['model']['k'], config['model']['temperature']
        )
        log(f'top k decoding took: {time.time() - start_time}', level='debug')
    log(f'time for predictions: {time.time() - start_time}', level='debug')

    # convert tokens to text
    preds = []
    preds += ids_to_toks(tgt_pred, tokenizer, sort=False)
    return ' '.join(preds[0])

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
    tgt_mask = Variable(torch.LongTensor(
        [
            [1] for i in range(src_input.size(0))
        ]
    ))

    if CUDA:
        tgt_input = tgt_input.cuda()
        tgt_mask = tgt_mask.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logits, _, decoder_states = model(src_input, tgt_input, srcmask, srclens,
            aux_input, auxmask, auxlens, tgt_mask)
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
        next_preds = Variable(torch.from_numpy(sampled_indices))
        prev_mask = tgt_mask.data.cpu().numpy()[:,-1]
        next_mask = [[0] if cur == [stop_id] or prev == [0] else [1] 
            for cur, prev in zip(sampled_indices, prev_mask)]
        next_mask = Variable(torch.from_numpy(np.array(next_mask)))
        if CUDA:
            next_preds = next_preds.cuda()
            next_mask = next_mask.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)
        tgt_mask = torch.cat((tgt_mask, next_mask), dim=1)

    # return decoder_states (top_k sampling)
    return tgt_input, decoder_states

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
                    aux_input, auxmask, auxlens)
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

    
