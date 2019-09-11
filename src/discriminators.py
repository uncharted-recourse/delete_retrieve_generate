
import torch.nn as nn
from pytorch_transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel#, XLNetLMHeadModel, TransfoXLLMHeadModel
import torch.optim as optim

# L_z - aligns latent state z 
    # inputs = final hidden states of encoder for each style s

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

class ConvNet(nn.Module):
    """ wrapper for a relatively simple CNN 

        num_classes = number of classes CNN should discriminate between
        num_channels = list of output channels for each conv layer
        kernel_sizes = list of kernel sizes for each conv layer
        conv_dim = dimension of CNN input, can be 1 or 2
        """
    def __init__(self, num_classes, num_channels, kernel_sizes, conv_dim):
        
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        inp_channels = [1]
        inp_channels.append(num_channels[1:])
        
        if conv_dim == 1:
            self.conv_layers = [nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernels, stride=1, padding=kernels // 2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) for in_c, out_c, kernels in zip(inp_channels, num_channels, kernel_sizes)]
        elif conv_dim == 2: 
            self.conv_layers = [nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernels, stride=1, padding=kernels // 2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) for in_c, out_c, kernels in zip(inp_channels, num_channels, kernel_sizes)]
        else:
            raise NotImplementedError("Convolution dimension must be one or two")

    def forward(self, content):
        for layer in self.conv_layers:
            content = layer(content)
        content = content.reshape(content.size(0), -1)

        # define fully connected layer shape based on size of input
        fc_input_dim = num_channels[-1]
        for dim in content.size()[1:]:
            fc_input_dim *= dim
        fc = nn.Linear(fc_input_dim, self.num_classes)

        return fc(content)

def define_discriminators(n_styles, working_dir, lr, optimizer_type, scheduler_type):
    """ Defines z and style discriminators, optimizers, and schedulers"""

    # z discriminator discriminates between z encoder final hidden state (lstm) 
    # or encoder output (transformer) from n styles
    z_discriminator = ConvNet(
        n_styles = n_styles,
        num_channels = [2,4], 
        kernel_sizes = [5,5], 
        conv_dim = 1
        )

    # style disciminators discriminate between hidden states (lstm) or output state
    # (transformer) from generated example and hidden states (lstm) or output state
    # (transformer) from teacher-forced example, for each style separately
    s_discriminators = [ConvNet(2, 
        num_channels = [2,4], 
        kernel_sizes_s = [5,5], 
        conv_dim = 2
        ) for _ in range(0, n_styles)]

    z_discriminator, _ = models.attempt_load_model(
        model=z_discriminator,
        checkpoint_dir=working_dir)
    s_discriminators = [models.attempt_load_model(
        model=s_discriminator,
        checkpoint_dir=working_dir)[0] for s_discriminator in s_discriminators]
    if CUDA:
        z_discriminator = z_discriminator.cuda()
        s_discriminators = [s_discriminator.cuda() for s_discriminator in s_discriminators]

    # define learning rates and scheduler
    # we've already checked for not implemented errors above 
    params = [z_discriminator.parameters()]
    for s_discriminator in s_discriminators:
        params.append(s_discriminator.parameters())
    if optimizer_type == 'adam':
        d_optimizers = [optim.Adam(param, lr=lr) for param in params]
    else: 
        d_optimizers = [optim.SGD(param, lr=lr) for param in params]
    if scheduler_type == 'plateau':
        d_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 'min') for d_optimizer in d_optimizers]
    else:
        d_schedulers = [optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 
            base_lr = lr,  
            max_lr = 10 * lr
        ) for d_optimizer in d_optimizers]
    
    return z_discriminator, s_discriminators, d_optimizers, d_schedulers
    
# LM Discrim

# Step 1: NLLL of L_lm_x and L_lm_y
# Step 2: minimize reconstruction loss + perplexity according to L_lm_x and L_lm_y

class LanguageModel(nn.Module):
    def __init__(self, tgt_vocab_size, join_method = 'add', finetune = False, model_name = 'gpt2', cache_dir = None):
        super(LanguageModel, self).__init__()

        models = {
            'gpt': OpenAIGPTLMHeadModel, 
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
        self.language_model = models[model_name].from_pretrained(model_weights[model_name], 
            cache_dir=cache_dir
        )

        # resize token embeddings if vocabulary has been augmented with special tokens
        self.language_model.resize_token_embeddings(tgt_vocab_size)

        # finetune if desired
        if not finetune:
            for param in self.language_model.parameters():
                param.requires_grad = False

    def forward(self, input_tgt):
        return self.language_model(input_tgt)[0]
