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
import random


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
working_dir = os.path.join("checkpoints", config['data']['working_dir'])
vocab_dir = os.path.join("checkpoints", config['data']['vocab_dir'])

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

config_path = os.path.join(working_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

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
input_lines_src = [l.strip().split() for l in open(config['data']['src'], 'r')]
input_lines_tgt = [l.strip().split()  for l in open(config['data']['tgt'], 'r')]
input_lines_src_test = [l.strip().split()  for l in open(config['data']['src_test'], 'r')]
input_lines_tgt_test = [l.strip().split()  for l in open(config['data']['tgt_test'], 'r')]

src, tgt = data.read_nmt_data(
   src_lines=input_lines_src,
   tgt_lines=input_lines_tgt,
   config=config,
   cache_dir=vocab_dir
)
src_test, tgt_test = data.read_nmt_data(
    src_lines=input_lines_src_test,
    tgt_lines=input_lines_tgt_test,
    config=config,
    train_src=src,
    train_tgt=tgt,
    cache_dir=vocab_dir
)
logging.info('...done!')

# grab important params from config
batch_size = config['data']['batch_size']
max_length = config['data']['max_len']
src_vocab_size = tgt_vocab_size = len(src['tokenizer'])
padding_id = data.get_padding_id(src['tokenizer'])
assert padding_id == src['tokenizer'].vocab_size + 1
torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])
writer = SummaryWriter(working_dir)

# define and load model
model = models.FusedSeqModel(
   src_vocab_size=src_vocab_size,
   tgt_vocab_size=tgt_vocab_size,
   pad_id_src=padding_id,
   pad_id_tgt=padding_id,
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
if config['training']['optimizer'] == 'adam':
    lr = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['learning_rate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommended for this task")

scheduler_name = config['training']['scheduler']
# reduce learning rate by a factor of 10 after plateau of 10 epochs
if scheduler_name == 'plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
elif scheduler_name == 'cyclic':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        base_lr = lr,  
        max_lr = 10 * lr
    )
else:
    raise NotImplementedError("Learning scheduler not recommended for this task")

# main training loop
epoch_loss = []
start_since_last_report = time.time()
words_since_last_report = 0
losses_since_last_report = []
best_metric = 0.0
best_epoch = 0
cur_metric = 0.0 # log perplexity or BLEU
num_batches = len(src['content']) / batch_size

STEP = 0
for epoch in range(start_epoch, config['training']['epochs']):
    epoch_start_time = time.time()
    if cur_metric > best_metric:
        # rm old checkpoint
        for ckpt_path in glob.glob(working_dir + '/model.*'):
            os.system("rm %s" % ckpt_path)
        # replace with new checkpoint
        torch.save(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)

        best_metric = cur_metric
        best_epoch = epoch - 1

    losses = []
    for i in range(0, len(src['content']), batch_size):

        if args.overfit:
            i = 50

        batch_idx = i / batch_size

        # calculate loss
        optimizer.zero_grad()
        loss_crit = config['training']['loss_criterion']
        train_loss, _ = evaluation.calculate_loss(src, tgt, i, batch_size, max_length, 
            config['model']['model_type'], loss_crit=loss_crit, config['training']['bt_ratio'])
        loss_item = train_loss.item() if loss_crit == 'cross_entropy' else -train_loss.item()
        losses.append(loss_item)
        losses_since_last_report.append(loss_item)
        epoch_loss.append(loss_item)
        train_loss.backward()

        # write information to tensorboard
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])
        writer.add_scalar('stats/grad_norm', norm, STEP)
        optimizer.step()

        if scheduler_name == 'cyclic':
            writer.add_scalar('stats/lr', scheduler.get_lr(), STEP)

        if args.overfit or batch_idx % config['training']['batches_per_report'] == 0:

            s = float(time.time() - start_since_last_report)
            wps = (batch_size * config['training']['batches_per_report']) / s
            avg_loss = np.mean(losses_since_last_report)
            info = (epoch, batch_idx, num_batches, wps, avg_loss, cur_metric)
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
    model.eval()
    dev_loss, mean_entropy = evaluation.evaluate_lpp(
            model, src_test, tgt_test, config)

    writer.add_scalar('eval/loss', dev_loss, epoch)
    writer.add_scalar('stats/mean_entropy', mean_entropy, epoch)
    if scheduler_name == 'plateau':
        for param_group in optimizer.param_groups:
            writer.add_scalar('stats/lr', param_group['lr'], epoch)
        scheduler.step(dev_loss)

    # write predictions and ground truths to checkpoint dir
    if args.bleu and epoch >= config['training'].get('inference_start_epoch', 1):
        
        cur_metric, edit_distance, inputs, preds, golds, auxs = evaluation.inference_metrics(
            model, src_test, tgt_test, config)

        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        writer.add_scalar('eval/edit_distance', edit_distance, epoch)
        writer.add_scalar('eval/bleu', cur_metric, epoch)

    else:
        cur_metric = dev_loss
   
    model.train()
    logging.info('METRIC: %s. TIME: %.2fs CHECKPOINTING...' % (cur_metric, (time.time() - start)))
    avg_loss = np.mean(epoch_loss)
    epoch_loss = []
    logging.info(f'Epoch took {time.time() - epoch_start_time} seconds')
writer.close()

