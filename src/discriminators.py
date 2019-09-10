
import torch.nn as nn
from pytorch_transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel#, XLNetLMHeadModel, TransfoXLLMHeadModel
import src.models as models
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
    z_discriminator = ConvNet(
        n_styles = n_styles,
        num_channels = [2,4], 
        kernel_sizes = [5,5], 
        conv_dim = 1
        )
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
        checkpoint_dir=working_dir) for s_discriminator in s_discriminators][0]]
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
    
class Discriminator(nn.Module):
    """ discriminates between encoder final hidden state / output state, and, for each style, 
        between decoder hidden states / decoder output 

        n_styles = number of different styles for discriminator to discriminate against
        num_channels_z = list of output channels for each conv layer in z discriminator
        kernel_sizes_z = list of kernel sizes for each conv layer in z discriminator
        num_channels_s = list of output channels for each conv layer in style discriminator
        kernel_sizes_s = list of kernel sizes for each conv layer in style discriminator
    """

    def __init__(self, n_styles, num_channels_z, kernel_sizes_z, num_channels_s = None, kernel_sizes_s = None):
        super(Discriminator, self).__init__()

        if num_channels_s is None:
            num_channels_s = num_channels_z
        if kernel_sizes_s is None:
            kernel_sizes_s = kernel_sizes_z
        self.n_styles = n_styles

        # z discriminator discriminates between z encoder final hidden state (lstm) 
        # or encoder output (transformer) from n styles
        self.z_discriminator = ConvNet(n_styles, 
            num_channels_z, 
            kernel_sizes_z, 
            1
        )

        # style disciminators discriminate between hidden states (lstm) or output state
        # (transformer) from generated example and hidden states (lstm) or output state
        # (transformer) from teacher-forced example, for each style separately
        self.style_discriminators = [ConvNet(2, 
            num_channels_s, 
            kernel_sizes_s, 
            2 
        ) for i in range(0, n_styles)]

    def forward(self, encoder_outputs, tf_decoder_states, soft_decoder_states):
        """ encoder_outputs = encoder outputs
            tf_decoder_states: unrolled decoder hidden states from D(z, s_i) with x_i (real)
            soft_decoder_states: unrolled decoder hidden states from D(z, s_j) (fake) 
            """

        z_output = self.z_discriminator.forward(encoder_output)

        # split decoder states according to n_styles
        style_batch_size = (tf_decoder_states.size(0) // self.n_styles)
        
        # pass tf_decoder_states and soft_decoder_states to each style discriminator
        s_outputs = []
        for i in n_styles:
            tf_states = tf_decoder_states[i * style_batch_size:(i+1) * style_batch_size]
            soft_states = soft_decoder_states[i * style_batch_size:(i+1) * style_batch_size]
            s_outputs.append(self.style_discriminators[i].forward(torch.cat((tf_states, soft_states), dim=0)))

        return z_output, s_outputs
    
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