import torch
import torch.nn as nn
from pytorch_transformers import OpenAIGPTModel, OpenAIGPTLMHeadModel, GPT2LMHeadModel#, XLNetLMHeadModel, TransfoXLLMHeadModel
import torch.optim as optim
from src import models
from src.cuda import CUDA
import logging
import numpy as np
import os
from utils.log_func import get_log_func

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

# L_s - discriminate between G(z, s) and G(z,s) ??
    # inputs = final unrolled hidden states of decoder on generated x_s and 
    # hidden states on x (teacher forced according to actual words)

# Overall - 
    # L_d = (L_z + sum_{s} L_s)
    # minimize L = L_rec - lambda * (L_z + sum_{s} L_s)

# Training loop

    # Divide minibatch into s pieces representing each style

    # Step 1: minimize discriminator loss L_d (update Discrim)
        # Encode sub-minibatch for each s
        # D(x,s) for each sub_minibatch s
        # D(z,s) for each sub_minibatch s
    # Step 2: minimize L (update Enc, Dec)

class CNNSequentialBlock(nn.Module):
    """ 
        Defines a sequential block for the CNN: convolution, batch norm, relu, maxpooling
        Necessary for pytorch to recognize parameters in list

        in_channels = input channels for conv layer
        out_channels = output channels for conv layer
        kernel_size = kernel size for conv layer
        conv_dim = dimension of conv input, can be 1 or 2 (also applies to maxpooling layer)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, conv_dim, pooling_stride):

        super(CNNSequentialBlock, self).__init__()

        if conv_dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.max_pool = nn.MaxPool1d(kernel_size=2, stride=pooling_stride)
            self.batch_norm = nn.BatchNorm1d(out_channels)
        elif conv_dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            self.batch_norm = nn.BatchNorm2d(out_channels)
        else:
            raise NotImplementedError("Convolution dimension must be one or two")
        self.relu = nn.ReLU()

    def forward(self, content):
        out = self.conv(content)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out

class ConvNet(nn.Module):
    """ wrapper for a relatively simple CNN 

        num_classes = number of classes CNN should discriminate between
        num_channels = list of output channels for each conv layer. input channels to first layer = max_length
        kernel_sizes = list of kernel sizes for each conv layer
        conv_dim = dimension of CNN input, can be 1 or 2
        """
    def __init__(self, num_classes, num_channels, kernel_sizes, conv_dim, pooling_stride, max_length, hidden_dim):
        
        super(ConvNet, self).__init__()

        inp_channels = [max_length]
        inp_channels += num_channels[:-1]
        layers = [CNNSequentialBlock(in_c, out_c, kernels, conv_dim, pooling_stride) for 
            in_c, out_c, kernels in zip(inp_channels, num_channels, kernel_sizes)]
        self.conv_blocks = nn.Sequential(*layers)
        
        # construct fully connected layer based on reduction dimensions
        reduce_factor = pooling_stride ** len(layers) 
        fc_in_dim = int(num_channels[-1] * (hidden_dim // reduce_factor))
        self.fc = nn.Linear(fc_in_dim, num_classes)
 
    def forward(self, content):
        out = self.conv_blocks(content)
        # resize appropriately for fully connected layer
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    # returns trainable params, untrainable params
    def count_params(self):
        n_trainable_params = 0
        n_untrainable_params = 0
        for param in self.parameters():
            if param.requires_grad:
                n_trainable_params += np.prod(param.data.cpu().numpy().shape)
            else:
                n_untrainable_params += np.prod(param.data.cpu().numpy().shape)
        return n_trainable_params, n_untrainable_params

def define_discriminators(n_styles, max_length_s, hidden_dim, working_dir, lr, optimizer_type, scheduler_type):
    """ Defines style discriminators, optimizers, and schedulers"""

    # z discriminator discriminates between z encoder final hidden state (lstm) 
    # or encoder output (transformer) from n styles
    # z_discriminator = ConvNet(
    #     num_classes = n_styles,
    #     num_channels = [2,4], 
    #     kernel_sizes = [5,5], 
    #     conv_dim = 1, 
    #     pooling_stride = 4,
    #     max_length = 1, 
    #     hidden_dim = hidden_dim
    #     )

    # style disciminators discriminate between hidden states (lstm) or output state
    # (transformer) from generated example and hidden states (lstm) or output state
    # (transformer) from teacher-forced example, for each style separately
    s_discriminators = [ConvNet(
        num_classes = 2, 
        num_channels = [100,200], 
        kernel_sizes = [32,64], 
        conv_dim = 1, 
        pooling_stride = 4,
        max_length = max_length_s,
        hidden_dim = hidden_dim
        ) for _ in range(0, n_styles)]
    # trainable, untrainable = z_discriminator.count_params()
    # logging.info(f'Z discriminator has {trainable} trainable params and {untrainable} untrainable params')
    trainable, untrainable = s_discriminators[0].count_params()
    logging.info(f'Style discriminators have {trainable} trainable params and {untrainable} untrainable params')
    
    # z_discriminator, _ = models.attempt_load_model(
    #     model=z_discriminator,
    #     checkpoint_dir=working_dir)
    s_discriminators = [models.attempt_load_model(
        model=s_discriminator,
        checkpoint_dir=working_dir,
        model_type=f's_discriminator_{idx}')[0] for idx, s_discriminator in enumerate(s_discriminators)]
    if CUDA:
        # z_discriminator = z_discriminator.cuda()
        s_discriminators = [s_discriminator.cuda() for s_discriminator in s_discriminators]
    
    # define learning rates and scheduler
    # we've already checked for not implemented errors above 
    # params = [z_discriminator.parameters()]
    # for s_discriminator in s_discriminators:
    #     params.append(s_discriminator.parameters())
    if optimizer_type == 'adam':
        d_optimizers = [optim.Adam(s_discriminator.parameters(), lr=lr) for s_discriminator in s_discriminators]
    else: 
        d_optimizers = [optim.SGD(s_discriminator.parameters(), lr=lr) for s_discriminator in s_discriminators]
    if scheduler_type == 'plateau':
        d_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 'min') for d_optimizer in d_optimizers]
    else:
        d_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 
            base_lr = lr,  
            max_lr = 10 * lr
        ) for d_optimizer in d_optimizers]
    
    return s_discriminators, d_optimizers, d_schedulers
    
# LM Discrim

# Step 1: NLLL of L_lm_x and L_lm_y
# Step 2: minimize reconstruction loss + perplexity according to L_lm_x and L_lm_y

class LanguageModel(nn.Module):
    def __init__(self, tgt_vocab_size, join_method = 'add', finetune = False, model_name = 'gpt2', cache_dir = None):
        super(LanguageModel, self).__init__()

        models = {
            'gpt': OpenAIGPTLMHeadModelAdversarial, 
            'gpt2': GPT2LMHeadModel, 
            #'xlnet': XLNetLMHeadModel,
            #'transformerxl': TransfoXLLMHeadModel
        }
        model_weights = {
            'gpt': 'openai-gpt', 
            'gpt2': 'gpt2', 
            #'xlnet': 'xlnet-base-cased',
            #'transformerxl': 'transfo-xl-wt103'
        }

        if model_name not in models.keys():
            raise Exception("Language model must be one of 'gpt', 'gpt2'")#, 'xlnet', 'transformerxl'")
    
        # !! assume that language model and seq2seq are using same tokenization !!
        self.lang_model = models[model_name].from_pretrained(model_weights[model_name], 
            cache_dir=cache_dir
        )

        # resize token embeddings if vocabulary has been augmented with special tokens
        self.lang_model.resize_token_embeddings(tgt_vocab_size)
        # finetune if desired

        if not finetune:
            for param in self.lang_model.parameters():
                param.requires_grad = False

    def forward(self, input_tgt, attention_mask = None):
        return self.lang_model(input_tgt, attention_mask = attention_mask)[0]

class OpenAIGPTModelAdversarial(OpenAIGPTModel):
    """ this class inherits from OpenAIGPTModel, but supports either discrete token ids as lookups to embedding layer 
        or continuous distributions over tokens to matmul with embedding weights"""

    def __init__(self, *args):
        super(OpenAIGPTModelAdversarial, self).__init__(*args)

    def _resize_token_embeddings(self, *args):
        super(OpenAIGPTModelAdversarial, self)._resize_token_embeddings(*args)

    def _prune_heads(self, *args):
        super(OpenAIGPTModelAdversarial, self)._prune_heads(*args)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # full documentation of this function can be found here 
        # https://huggingface.co/pytorch-transformers/_modules/pytorch_transformers/modeling_openai.html#OpenAIGPTModel

        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        input_shape = input_ids.size()

        # EDITED: multiply input_ids by embedding weights if input is probability distribution
        if len(input_shape) == 3: # batch_size, seq_length, vocab_size
            inputs_embeds = torch.matmul(input_ids, self.tokens_embed.weight)
        else:
            #input_ids = input_ids.view(-1, input_ids.size(-1))
            inputs_embeds = self.tokens_embed(input_ids)

        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
 
        output_shape = inputs_embeds.shape
        #output_shape = input_shape + (hidden_states.size(-1),)
        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, attention_mask, head_mask[i])
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        outputs = (hidden_states.view(*output_shape),)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (all hidden states), (all attentions)

class OpenAIGPTLMHeadModelAdversarial(OpenAIGPTLMHeadModel):
    """ this class inherits from OpenAIGPTLMHeadModel, but supports either discrete token ids as lookups to embedding layer 
    or continuous distributions over tokens to matmul with embedding weights"""

    def __init__(self, config):
        super(OpenAIGPTLMHeadModelAdversarial, self).__init__(config)
        self.transformer = OpenAIGPTModelAdversarial(config)

    def tie_weights(self):
        super(OpenAIGPTLMHeadModelAdversarial, self).tie_weights()
    
    def forward(self, *args, **kwargs):
        return super(OpenAIGPTLMHeadModelAdversarial, self).forward(*args, **kwargs)
