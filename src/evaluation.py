import math
import numpy as np
import sys
from collections import Counter
from typing import List
import torch
from torch.autograd import Variable
import torch.nn as nn
import editdistance
import heapq
from src import data
from src.cuda import CUDA
from src.callbacks import mean_masked_entropy
import random
import time

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

def calculate_loss(src, tgt, config, i, batch_size, max_length, model_type, model, loss_crit = 'cross_entropy', bt_ratio = 1):
    
    use_src = random.random() < 0.5

    # get normal minibatch
    input_content, input_aux, output = data.minibatch(
        src, tgt, i, batch_size, max_length, model_type, use_src=use_src)
    input_lines_src, _, srclens, srcmask, _ = input_content
    input_ids_aux, _, auxlens, auxmask, _ = input_aux
    input_lines_tgt, output_lines_tgt, tgtlens, _, _ = output
    
    decoder_logit, decoder_probs = model(
        input_lines_src, input_lines_tgt, srcmask, srclens,
        input_ids_aux, auxlens, auxmask)
    
    # calculate loss on two minibatches separately, weight losses w/ ratio
    weight_mask = torch.ones(len(src['tokenizer']))
    if CUDA:
        weight_mask = weight_mask.cuda()
    padding_id = data.get_padding_id(src['tokenizer'])
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
            decoder_logit.contiguous().view(-1, len(src['tokenizer'])),
            output_lines_tgt.view(-1)
        )
    else:
        # calculate lb expected bleu loss with max_order ngrams of 4 independent of ngram_range in config
        loss = bleu(decoder_probs, output_lines_tgt.cpu(), 
            torch.LongTensor([max_length] * batch_size),
            tgtlens , smooth=True)[0]

    # mean entropy
    mean_entropy = mean_masked_entropy(decoder_probs.data.cpu().numpy(), weight_mask.data.cpu().numpy, padding_id)

    # get backtranslation minibatch
    if bt_ratio > 0:
        bt_input_content, bt_input_aux, bt_output = data.back_translation_minibatch(
            src, tgt, config, i, batch_size, max_length, model,  model_type, use_src=use_src)
        bt_input_lines_src, _, bt_srclens, bt_srcmask, _ = bt_input_content
        bt_input_ids_aux, _, bt_auxlens, bt_auxmask, _ = bt_input_aux
        bt_input_lines_tgt, bt_output_lines_tgt, bt_tgtlens, _, _ = bt_output
        
        bt_decoder_logit, bt_decoder_probs = model(
            bt_input_lines_src, bt_input_lines_tgt, bt_srcmask, bt_srclens,
            bt_input_ids_aux, bt_auxlens, bt_auxmask)
        
        # calculate loss
        if loss_crit == 'cross_entropy':
            bt_loss = loss_criterion(
                bt_decoder_logit.contiguous().view(-1, len(src['tokenizer'])),
                bt_output_lines_tgt.view(-1)
            )
        else:
            bt_loss = bleu(bt_decoder_probs, bt_output_lines_tgt.cpu(), 
                torch.LongTensor([max_length] * batch_size),
                bt_tgtlens, smooth=True)[0]

        # combine losses
        loss = (bt_ratio * bt_loss + loss) / 2

        # mean entropy
        bt_mean_entropy = mean_masked_entropy(bt_decoder_probs.data.cpu().numpy(), weight_mask.data.cpu().numpy, padding_id)
        mean_entropy = (bt_ratio * bt_mean_entropy + mean_entropy) / 2
 
    # return combined loss and combined mean entropy
    return loss, mean_entropy

def decode_minibatch_greedy(max_len, start_id, model, src_input, srclens, srcmask,
        aux_input, auxlens, auxmask):
    """ argmax decoding """
    # Initialize target with <s> for every sentence
    tgt_input = Variable(torch.LongTensor(
        [
            [start_id] for i in range(src_input.size(0))
        ]
    ))

    if CUDA:
        tgt_input = tgt_input.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logit, word_probs = model(src_input, tgt_input, srcmask, srclens,
            aux_input, auxmask, auxlens)
        decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
        # select the predicted "next" tokens, attach to target-side inputs
        next_preds = Variable(torch.from_numpy(decoder_argmax[:, -1]))
        if CUDA:
            next_preds = next_preds.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

    return tgt_input

# convert seqs to tokens
def ids_to_toks(tok_seqs, tokenizer, sort = True, indices = None):
    #out = []
    # take off the gpu
    tok_seqs = tok_seqs.cpu().numpy()
    # convert to toks, delete any special tokens (bos, eos, pad)
    start_id = data.get_start_id(tokenizer)
    stop_id = data.get_stop_id(tokenizer)
    tok_seqs = [line[1:] if line[0] == start_id else line for line in tok_seqs]
    tok_seqs = [np.split(line, np.where(line == stop_id)[0])[0] for line in tok_seqs]
    tok_seqs = [tokenizer.decode(line) for line in tok_seqs]
    #     toks = tokenizer.decode(line)
    #     toks = toks.split('<s>')
    #     toks = toks[0] if len(toks) == 1 else toks[1]
    #     toks = toks.split('</s>')[0]
    #     out.append(toks)
    # unsort
    if sort:
         return data.unsort(tok_seqs, indices)
    else:
         return tok_seqs

def generate_sequences(tokenizer, model, config, start_id, stop_id, input_content, input_aux, output):
    input_lines_src, output_lines_src, srclens, srcmask, indices = input_content
    input_ids_aux, _, auxlens, auxmask, _ = input_aux
    input_lines_tgt, output_lines_tgt, _, _, _ = output

    # decode dataset with greedy, beam search, or top k
    start_time = time.time()
    if config['model']['decode'] == 'greedy':
        tgt_pred = decode_minibatch_greedy(
            config['data']['max_len'], start_id, 
            model, input_lines_src, srclens, srcmask,
            input_ids_aux, auxlens, auxmask)
        log(f'greedy search decoding took: {time.time() - start_time}', level='debug')
    elif config['model']['decode'] == 'beam_search':
        start_time = time.time()
        if config['model']['model_type'] == 'delete_retrieve':
            tgt_pred = torch.stack([beam_search_decode(
                model, i, [i_l], i_m, a, [a_l], a_m,
                start_id, stop_id,
                config['data']['max_len'], config['model']['beam_width']) for 
                i, i_l, i_m, a, a_l, a_m in zip(input_lines_src, srclens, srcmask,
                input_ids_aux, auxlens, auxmask)])
        elif config['model']['model_type'] == 'delete':
            input_ids_aux = input_ids_aux.unsqueeze(1)
            tgt_pred = torch.stack([beam_search_decode(
                model, i, [i_l], i_m, a, None, None,
                start_id, stop_id,
                config['data']['max_len'], config['model']['beam_width']) for 
                i, i_l, i_m, a in zip(input_lines_src, srclens, srcmask,
                input_ids_aux)])
        log(f'beam search decoding took: {time.time() - start_time}', level='debug')
    elif config['model']['decode'] == 'top_k':
        if config['model']['model_type'] == 'delete_retrieve':
            tgt_pred = torch.stack([top_k_decode(
                model, i, [i_l], i_m, a, [a_l], a_m,
                start_id, stop_id,
                config['data']['max_len'], config['model']['k'], config['model']['temperature']) for 
                i, i_l, i_m, a, a_l, a_m in zip(input_lines_src, srclens, srcmask,
                input_ids_aux, auxlens, auxmask)])
        elif config['model']['model_type'] == 'delete':
            input_ids_aux = input_ids_aux.unsqueeze(1)
            tgt_pred = torch.stack([top_k_decode(
                model, i, [i_l], i_m, a, None, None,
                start_id, stop_id,
                config['data']['max_len'], config['model']['k'], config['model']['temperature']) for 
                i, i_l, i_m, a in zip(input_lines_src, srclens, srcmask,
                input_ids_aux)])
        log(f'top k decoding took: {time.time() - start_time}', level='debug')
    else:
        raise Exception('Decoding method must be one of greedy, beam_search, top_k')

    return tgt_pred

def decode_dataset(model, src, tgt, config):
    """Evaluate model."""
    inputs = []
    preds = []
    auxs = []
    ground_truths = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()

        # get batch
        input_content, input_aux, output, = data.minibatch(
            src, tgt, j, 
            config['data']['batch_size'], 
            config['data']['max_len'], 
            config['model']['model_type'],
            is_test=True)
        input_lines_src, output_lines_src, srclens, srcmask, indices = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output

        # generate sequences according to decoding strategy
        tokenizer = src['tokenizer']
        tgt_pred = generate_sequences(
            tokenizer,
            model, 
            config,
            data.get_start_id(tokenizer),
            data.get_stop_id(tokenizer),
            input_content,
            input_aux,
            output
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


def inference_metrics(model, src, tgt, config):
    """ decode and evaluate bleu """
    inputs, preds, ground_truths, auxs = decode_dataset(
        model, src, tgt, config)
    bleu = get_bleu(preds, ground_truths)
    edit_distance = get_edit_distance(preds, ground_truths)

    inputs = [''.join(seq) for seq in inputs]
    preds = [''.join(seq) for seq in preds]
    ground_truths = [''.join(seq) for seq in ground_truths]
    auxs = [''.join(seq) for seq in auxs]

    return bleu, edit_distance, inputs, preds, ground_truths, auxs


def evaluate_lpp(model, src, tgt, config):
    """ evaluate log perplexity WITHOUT decoding
        (i.e., with teacher forcing)
    """
    losses = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()

        loss_crit = config['training']['loss_criterion']
        combined_loss, combined_mean_entropy = calculate_loss(src, tgt, config, j, config['data']['batch_size'], config['data']['max_len'], 
            config['model']['model_type'], model, loss_crit=loss_crit, bt_ratio=config['training']['bt_ratio'])

        loss_item = combined_loss.item() if loss_crit == 'cross_entropy' else -combined_loss.item()
        losses.append(loss_item)

    return np.mean(losses), combined_mean_entropy

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
                model, i, [i_l], i_m, a, [a_l], a_m,
                start_id, stop_id,
                max_len, config['model']['beam_width']) for 
                i, i_l, i_m, a, a_l, a_m in zip(content, content_length, content_mask,
                attributes, attributes_len, attributes_mask)])
        elif config['model']['model_type'] == 'delete':
            input_ids_aux = input_ids_aux.unsqueeze(1)
            tgt_pred = torch.stack([beam_search_decode(
                model, i, [i_l], i_m, a, None, None,
                start_id, stop_id,
                max_len, config['model']['beam_width']) for 
                i, i_l, i_m, a in zip(content, content_length, content_mask,
                attributes)])
        log(f'beam search decoding took: {time.time() - start_time}', level='debug')
    elif config['model']['decode'] == 'top_k':
        start_time = time.time()
        if config['model']['model_type'] == 'delete_retrieve':
            tgt_pred = torch.stack([top_k_decode(
                model, i, [i_l], i_m, a, [a_l], a_m,
                start_id, stop_id,
                max_len, config['model']['k'], config['model']['temperature']) for 
                i, i_l, i_m, a, a_l, a_m in zip(content, content_length, content_mask,
                attributes, attributes_len, attributes_mask)])
        elif config['model']['model_type'] == 'delete':
            input_ids_aux = input_ids_aux.unsqueeze(1)
            tgt_pred = torch.stack([top_k_decode(
                model, i, [i_l], i_m, a, None, None,
                start_id, stop_id,
                max_len, config['model']['k'], config['model']['temperature']) for 
                i, i_l, i_m, a in zip(content, content_length, content_mask,
                attributes)])
        log(f'top k decoding took: {time.time() - start_time}', level='debug')
    log(f'time for predictions: {time.time() - start_time}', level='debug')

    # convert tokens to text
    preds = []
    preds += ids_to_toks(tgt_pred, tokenizer, sort=False)
    return ' '.join(preds[0])

def sample_softmax(logits, temperature=1.0, num_samples=1):
    exps = np.exp((logits - np.max(logits)) / temperature)
    probs = exps / np.sum(exps)
    return np.random.multinomial(num_samples, probs, 1)

def get_next_token_scores(model, src_input, tgt_input, srcmask, srclen, 
                        aux_input, auxmask, auxlen):
    
    # get tensors in correct shape for prediction
    src_input = src_input.unsqueeze(0)
    tgt_input = tgt_input.unsqueeze(0)
    srcmask = srcmask.unsqueeze(0)
    if auxmask is not None:
        auxmask = auxmask.unsqueeze(0)
    if CUDA:
        tgt_input = tgt_input.cuda()
    decoder_logit, word_probs = model(src_input, tgt_input, srcmask, srclen,
        aux_input, auxmask, auxlen)
    return decoder_logit[0, tgt_input.size()[1] - 1, :]

def top_k_decode(
    model: nn.Module,
    input_src: List[int] = None,
    srclen: int = None,
    srcmask: List[int] = None,
    input_aux: List[int] = None,
    auxlen: int = None,
    auxmask: List[int] = None,
    start_id: int = None,
    stop_id: int = None,
    max_seq_length: int = 50,
    k: int = 10,
    temperature=1.0,
    init_prefix=None,
    num_return=1,
    return_scores=False,
):
    init_seq = torch.tensor(init_prefix) if init_prefix else torch.tensor([start_id])
    output_seqs = []
    for _ in range(num_return):
        generated_seq = init_seq
        total_score = 0
        
        # exit condition: either hit max length or find stop character
        while (generated_seq[-1] != stop_id) and (generated_seq.size()[0] < max_seq_length):
            next_token_scores = get_next_token_scores(
                model, input_src, generated_seq, srcmask, srclen, 
                input_aux, auxmask, auxlen
            )
            next_token_scores = next_token_scores.data.cpu().numpy()
            
            # if k=1, do greedy sampling
            if k == 1:
                sampled_index = np.argmax(next_token_scores)
            
            # if k > 1, do softmax sampling over the top-k
            elif k:
                # grab last k (largest scores)
                top_ids = np.argsort(next_token_scores)[-k:]
                top_scores = next_token_scores[top_ids]
                ind = sample_softmax(top_scores, temperature=temperature)
                sampled_index = top_ids[ind]
            
            # if k is falsey (eg None), do softmax sampling over full vocab
            else:
                sampled_index = sample_softmax(
                    next_token_scores, temperature=temperature
                )
            total_score += next_token_scores[sampled_index]
            generated_seq = torch.cat((generated_seq, sampled_index), dim=0)
        output_seqs.append((total_score, generated_seq))
    
    # sort by score
    output_seqs = sorted(output_seqs, reverse=True)
    
    # strip scores if not being returned
    output_seqs = output_seqs if return_scores else [seq[1] for seq in output_seqs]
    if num_return == 1:
        return output_seqs[0]
    else:
        assert len(output_seqs) == num_return
        return output_seqs

class Beam(object):
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
    model: nn.Module,
    input_src: List[int] = None,
    srclen: int = None,
    srcmask: List[int] = None,
    input_aux: List[int] = None,
    auxlen: int = None,
    auxmask: List[int] = None,
    start_id: int = None,
    stop_id: int = None,
    max_seq_length: int = 50,
    beam_width: int = 10,
    init_prefix=None,
    num_return = 1,
    return_scores = False
):
    """ Beam search decoding method 
    
    Arguments:
        input_seq {List(int)} -- [Input sequence of integers]
    
    Keyword Arguments:
        beam_width {int} -- [Number of beams/sequences to use in search (higher can get better responses, but takes longer)] (default: {10})
    
    Returns:
        [(float, list[int])] -- sorted list of (score, sequence) pairs
        
    """
    num_return = num_return or beam_width
    prev_beam = Beam(beam_width)
    
    if init_prefix:
        prev_beam.add(1.0, False, torch.LongTensor(init_prefix))
    else:
        prev_beam.add(1.0, False, torch.LongTensor([start_id]))
    while True:
        curr_beam = Beam(beam_width)
        
        # Add complete sentences to the current beam, add more words to the rest
        for (prefix_score, complete, prefix) in prev_beam:
            if complete:
                curr_beam.add(prefix_score, True, prefix)
            else:
                # Get probability of each possible next word for the incomplete prefix.
                next_token_scores  = get_next_token_scores(
                    model, input_src, prefix, srcmask, srclen, 
                    input_aux, auxmask, auxlen
                )
                for next_id, next_score in enumerate(next_token_scores):
                    score = prefix_score + next_score
                    new_prefix = torch.cat((prefix, torch.LongTensor([next_id])), dim=0)
                    now_complete = next_id == stop_id or new_prefix.size()[0] >= max_seq_length
                    curr_beam.add(score, now_complete, new_prefix)
        
        # if all beams are completed, sort and return (score, seq) pairs
        if all([complete for _, complete, _ in curr_beam]):
            curr_beam = sorted(curr_beam, reverse=True)[:num_return]
            if return_scores:
                generated_seqs = [
                    (score, prefix) for score, complete, prefix in curr_beam
                ]
            else:
                generated_seqs = [prefix for score, complete, prefix in curr_beam]
            
            if num_return == 1:
                return generated_seqs[0]
            else:
                return generated_seqs

        prev_beam = curr_beam

    
