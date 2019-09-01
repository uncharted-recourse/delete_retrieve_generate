
import torch.nn as nn
from pytorch_transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel#, XLNetLMHeadModel, TransfoXLLMHeadModel

# L_z - aligns latent state z 
    # inputs = final hidden states of encoder for each style s

# L_s - discriminate between x_s and D(z,s) ??
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

        kernel_sizes = list of kernel sizes for each conv layer
        num_channels = list of output channels for each conv layer """
    def __init__(self, num_classes, num_channels, kernel_sizes, emb_dim, input_lens):
        
        super(ConvNet, self).__init__()

        inp_channels = [1]
        inp_channels.append(num_channels[1:])

        self.conv_layers = [nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernels, stride=1, padding=kernels // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) for in_c, out_c, kernels in zip(inp_channels, num_channels, kernel_sizes)]

        self.fc = nn.Linear(emb_dim * input_lens * num_channels[-1], num_classes)

    def forward(self, content):
        for layer in self.conv_layers:
            content = layer(content)
        content = content.reshape(content.size(0), -1)
        return self.fc(content)

class Discriminator(nn.Module):
    """ discriminates between hidden states produced at various stages of the encoder, decoder process""" 
    def __init__(self, n_styles, num_channels, kernel_sizes, emb_dim, input_lens):
        super(Discriminator, self).__init__()

        # z discriminator discriminates between z hidden state from n styles
        self.z_discriminator = ConvNet(n_styles, 
            num_channels, 
            kernel_sizes, 
            emb_dim, 
            input_lens
        )

        # style disciminators discriminate between hidden state from generated example
        # and hidden state from real example, for each style separately
        self.style_discriminators = [ConvNet(2, 
            num_channels, 
            kernel_sizes, 
            emb_dim, 
            input_lens
        ) for i in range(0, n_styles)]

    def forward(self, encoder_output, decoder_hiddens_real, real_style_idx, decoder_hiddens_fake, fake_style_idx):
        """ encoder_outputs = encoder outputs
            decoder_hiddens_real: unrolled decoder hidden states from D(z, s_i) with x_i (real)
            decoder_hiddens_fake: unrolled decoder hidden states from D(z, s_j) (fake) """

        z_output = self.z_discriminator.forward(encoder_output)
        real_output = self.style_discriminators[real_style_idx].forward(decoder_hiddens_real)
        fake_output = self.style_discriminators[fake_style_idx].forward(decoder_hiddens_fake)
        return z_output, real_output, fake_output

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