import sys

import json
import numpy as np
import logging
import argparse
import os
import time
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import src.evaluation as evaluation
from src.cuda import CUDA
import src.data as data
import src.models as models
import src.discriminators as discriminators
import src.callbacks as callbacks
import random

# maximum batch sizes from experimentation on GPU
MAX_BS = 64
MAX_BS_DISCRIM = 4

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument(
    "--bleu",
    help="do BLEU eval",
    action='store_true'
)
parser.add_argument(
    "--overfit", 
    help="train continuously on one batch of data",
    action='store_true'
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))

# save all checkpoint folders to checkpoint dir
working_dir, vocab_dir, lm_dir = callbacks.on_train_start(config)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % working_dir,
)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
filelog = logging.FileHandler('%s/train_log' % working_dir)
filelog.setFormatter(formatter)
logger = logging.getLogger('')
logger.addHandler(console)
logger.addHandler(filelog)
logger.setLevel(logging.INFO)

# read data
logging.info('Reading data ...')
n_styles = len(config['data']['train'])
train_data = data.read_nmt_data(
   n_styles = n_styles,
   config=config,
   cache_dir=vocab_dir
)
test_data = data.read_nmt_data(
    n_styles = n_styles,
    config=config,
    train_data=train_data,
    cache_dir=vocab_dir
)
logging.info('...done!')

# grab important params from config
batch_size = config['data']['batch_size']
max_length = config['data']['max_len']
src_vocab_size = tgt_vocab_size = len(train_data[0]['tokenizer'])
#torch.manual_seed(config['training']['random_seed'])
#np.random.seed(config['training']['random_seed'])
writer = SummaryWriter(working_dir)

# Early stopping callback
early_stopping = callbacks.EarlyStopping(increase_good=False)

# define and load model
model = models.FusedSeqModel(
   src_vocab_size=src_vocab_size,
   tgt_vocab_size=tgt_vocab_size,
   pad_id_src=train_data[0]['tokenizer'].pad_token_id,
   pad_id_tgt=train_data[0]['tokenizer'].pad_token_id,
   config=config,
)
trainable, untrainable = model.count_params()
logging.info(f'MODEL HAS {trainable} trainable params and {untrainable} untrainable params')
model, start_epoch = models.attempt_load_model(
    model=model,
    checkpoint_dir=working_dir)
if CUDA:
    model = model.cuda()

# define learning rate and scheduler
scheduler_name = config['training']['scheduler']
optimizer, scheduler = evaluation.define_optimizer_and_scheduler(config['training']['learning_rate'], 
    config['training']['optimizer'], scheduler_name, model, config['training']['weight_decay'])

# define discriminator model, optimizers, and schedulers if adverarial paradigm
if config['training']['discriminator_ratio'] > 0:
    hidden_dim = config['model']['tgt_hidden_dim'] if config['model']['decoder'] == 'lstm' else config['model']['emb_dim']
    s_discriminators, d_optimizers, d_schedulers = discriminators.define_discriminators(
        n_styles,
        max_length,
        hidden_dim,
        working_dir, 
        config['training']['discriminator_learning_rate'],
        config['training']['weight_decay'],
        config['training']['optimizer'], 
        scheduler_name)
    update_frequency = batch_size // MAX_BS_DISCRIM
    batch_size = MAX_BS_DISCRIM 
else:
    s_discriminators = None
    update_frequency = batch_size // MAX_BS
    batch_size = MAX_BS

# main training loop

# track auto-encoder metrics
epoch_loss = []
start_since_last_report = time.time()
words_since_last_report = 0
losses_since_last_report = []
best_metric = 0.0
best_epoch = 0
cur_metric = 0.0 # log perplexity or BLEU

# track discriminator metrics
epoch_losses_discrim = [[]]
losses_discrim = [[]]

# training loop params
assert batch_size >= n_styles, "Batch size must be greater than or equal to the number of styles"
sample_size = batch_size // n_styles * 2 # x2 because can use bigger batch size at eval w/out gradients
content_lengths = [len(style_corpus['content']) for style_corpus in train_data]

# if adversarial paradigm always need >= 2 styles in minibatch to compare decoder states
if config['training']['discriminator_ratio'] > 0:
    content_lengths.remove(max(content_lengths))
num_batches = max(content_lengths) / sample_size

STEP = 0
for epoch in range(start_epoch, config['training']['epochs']):
    epoch_start_time = time.time()
    if cur_metric > best_metric:
        # rm old checkpoints
        for ckpt_path in glob.glob(working_dir + '/model.*'):
            os.system("rm %s" % ckpt_path)
        for ckpt_path in glob.glob(working_dir + '/s_discriminator*'):
            os.system("rm %s" % ckpt_path)

        # replace with new checkpoint
        torch.save(model.state_dict(), working_dir + f'/model.{epoch}.ckpt')
        if s_discriminators is not None:
            [torch.save(s_discriminator.state_dict(), working_dir + f'/s_discriminator_{idx}.{epoch}.ckpt')
                for idx, s_discriminator in enumerate(s_discriminators)]

        best_metric = cur_metric
        best_epoch = epoch - 1

    losses = []
    idx = max(content_lengths)
    batch_idx = 0
    while idx > 0:

        if args.overfit:
            idx = 50
        batch_idx += 1
        
        # set dataset list, batch_idx, and sample size according to corpii that support current idx range
        # (i.e. take advantage of corpii that have more examples than smallest corpii)
        style_ids = [i for i, corpus in enumerate(train_data) if idx in range(len(corpus['data']) + 1)]
        idx, train_sample_size = evaluation.update_training_index(idx, len(style_ids), batch_size)

        # calculate loss
        loss_crit = config['training']['loss_criterion']
        train_loss, s_losses = evaluation.calculate_loss(train_data, style_ids, n_styles, config, idx, train_sample_size, max_length, 
            config['model']['model_type'], model, s_discriminators, loss_crit, bt_ratio = config['training']['bt_ratio'])

        loss_item = train_loss.item() if loss_crit == 'cross_entropy' else -train_loss.item()
        losses.append(loss_item)
        losses_since_last_report.append(loss_item)
        epoch_loss.append(loss_item)

        # update discriminator optimizer and schedulers
        if s_discriminators is not None:
            bp_t = time.time()
            [evaluation.backpropagation_step(l, opt, sched, scheduler_name, batch_idx, update_frequency, 
                retain_graph=True) for l, opt, sched in zip(s_losses, d_optimizers, d_schedulers)]
            logging.debug(f'backpropagation through discriminators took: {time.time() - bp_t} seconds')
    
            # write information to tensorboard
            norms = [nn.utils.clip_grad_norm_(d.parameters(), config['training']['max_norm']) for d in s_discriminators]
            [writer.add_scalar(f'stats/grad_norm_discriminator_{idx}', norm, STEP) for idx, norm in enumerate(norms)]
            [loss_discrim.append(loss.item()) for loss_discrim, loss in zip(losses_discrim, s_losses)]
            if args.overfit or batch_idx % config['training']['batches_per_report'] == 0:
                avg_losses = [np.mean(loss_discrim) for loss_discrim in losses_discrim]
                [writer.add_scalar(f'stats/loss_discriminator_{idx}', avg_loss, STEP) for idx, avg_loss in enumerate(avg_losses)]
                for loss_discrim in losses_discrim:
                    loss_discrim = []
    
        bp_t = time.time()
        evaluation.backpropagation_step(train_loss, optimizer, scheduler, scheduler_name, batch_idx, update_frequency, retain_graph = False)
        logging.debug(f'backpropagation through S2S took: {time.time() - bp_t} seconds')
        
        # write information to tensorboard
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])
        writer.add_scalar('stats/grad_norm', norm, STEP)

        if scheduler_name == 'cyclic' or scheduler_name == 'cosine':
            writer.add_scalar('stats/lr', scheduler.get_lr(), STEP)

        if args.overfit or batch_idx % config['training']['batches_per_report'] == 0:

            s = float(time.time() - start_since_last_report)
            wps = (batch_size * config['training']['batches_per_report']) / s
            avg_loss = np.mean(losses_since_last_report)
            info = (epoch, num_batches - idx / sample_size, num_batches, wps, avg_loss, cur_metric)
            writer.add_scalar('stats/WPS', wps, STEP)
            writer.add_scalar('stats/loss', avg_loss, STEP)
            logging.info('EPOCH: %s ITER: %s/%s WPS: %.2f LOSS: %.4f METRIC: %.4f' % info)
            start_since_last_report = time.time()
            words_since_last_report = 0
            losses_since_last_report = []

        # NO SAMPLING!! because weird train-vs-test data stuff would be a pain
        STEP += 1

    if args.overfit:
        continue
    logging.info('EPOCH %s COMPLETE. EVALUATING...' % epoch)

    # evaluate on dev set, update scheduler
    start = time.time()    
    with torch.no_grad():
        dev_loss, d_dev_losses = evaluation.evaluate_lpp(model, s_discriminators, test_data, sample_size, config)

    writer.add_scalar('eval/loss', dev_loss, epoch)
    if s_discriminators is not None:
        [writer.add_scalar(f'eval/loss_discriminator_style_{idx}', d_dev_loss, epoch) for idx, d_dev_loss in enumerate(d_dev_losses)]
    #writer.add_scalar('stats/mean_entropy', mean_entropy, epoch)
    
    if scheduler_name == 'plateau':
        for param_group in optimizer.param_groups:
            writer.add_scalar('stats/lr', param_group['lr'], epoch)
        scheduler.step(dev_loss)

        if s_discriminators is not None:
            [d_scheduler.step(d_loss) for d_scheduler, d_loss in zip(d_schedulers, d_dev_losses)]
            
    # write predictions and ground truths to checkpoint dir
    if args.bleu and epoch >= config['training'].get('inference_start_epoch', 0):
        
        num_samples = config['training']['num_samples']
        with torch.no_grad():
            bleus, edit_distances, inputs, preds, golds, auxs = evaluation.inference_metrics(
                model, s_discriminators, test_data, sample_size, num_samples, config)
        
        # metrics averaged over metric for each style
        mean_bleu = np.mean(bleus)
        edit_distance = np.mean(edit_distances)
        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        # write edit distance and bleu metrics separately for each style
        [writer.add_scalar(f'eval/edit_distance_target_style_{(i + 1) % n_styles}', e, epoch) for i, e in enumerate(edit_distances)]
        [writer.add_scalar(f'eval/bleu_target_style_{(i + 1) % n_styles}', c, epoch) for i, c in enumerate(bleus)]
        writer.add_scalar('eval/bleu', mean_bleu, epoch)
        logging.info('BLEU: %s. TIME: %.2fs CHECKPOINTING...' % (mean_bleu, (time.time() - start)))
    else:
        logging.info('DEV LOSS: %s. TIME: %.2fs CHECKPOINTING...' % (dev_loss, (time.time() - start)))
    cur_metric = dev_loss
   
    #early_stopping(dev_loss)
    avg_loss = np.mean(epoch_loss)
    epoch_loss = []
    logging.info(f'Epoch took {time.time() - epoch_start_time} seconds')
writer.close()

