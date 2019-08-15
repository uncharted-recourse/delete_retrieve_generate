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
import tensorflow_datasets as tfds
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='decode.log',
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger = logging.getLogger('')
logger.addHandler(console)
logger.setLevel(logging.INFO)


if config['data']['encoder'] is not None:
    print(f"Loading encoder from {config['data']['encoder']}")
    encoder = tfds.features.text.SubwordTextEncoder.load_from_file(config['data']['encoder'])
else:
    print("No encoder found. Exiting.")
    sys.exit(1)

attribute_vocab = config['data']['attribute_vocab']
decoded_results_with_scores = []
with open(attribute_vocab) as f:
  next(f)
  for line in f:
    fields = line.split()
    #print(fields)
    num_fields = len(fields)
    pre_score = float(fields[num_fields-2])
    post_score = float(fields[num_fields-1])
    #print(f"score: pre={pre_score}, post={post_score}")
    to_be_decoded = [int(s) for s in fields[:num_fields-3] if s.isdigit()]
    #print(to_be_decoded)
    decoded = encoder.decode(to_be_decoded)
    #attribute_words[index] = decoded
    decoded_results_with_scores.append([decoded, pre_score, post_score])
    
    
df = pd.DataFrame(decoded_results_with_scores, columns=['attributewords', 'pre_score', 'post_score'])
df['attributewords'] = df['attributewords'].str.replace(" ","")
df['attributewords'] = df['attributewords'].str.replace("'","")
df['attributewords'] = df['attributewords'].str.replace('"','')
df_filtered = df[df['attributewords'].map(len) > 0]
sorted_by_pre = df_filtered.sort_values(by = 'pre_score', ascending = False)
sorted_by_post = df_filtered.sort_values(by = 'post_score' , ascending = False)
sorted_by_pre.to_csv('pre_attribute_words.txt', sep='\t', index = False)
sorted_by_post.to_csv('post_attribute_words.txt', sep='\t', index = False)
    
