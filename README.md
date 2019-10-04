# Description

This repository implements an auto-encoder approach to style transfer named **Ventriloquist**. The model uses a TransformerXL architecture with relative position attention for the encoder and decoder modules and is fused with OpenAI's GPT language model for regularization. 

The **flask.dockerfile** in the main repository builds a docker image that launches a flask app with a translation-like interface. An example screenshot is shown below:

![api_example](https://github.com/NewKnowledge/delete_retrieve_generate/blob/jg/discriminator/screenshots/api_example.png)

Users can input the number of translations, the target style(s), and the decoding parameters k / temperature and observe how these choices effect the stylized tranlsations!

# Extensions from Original Implementation

This repository was originally forked from rpryzant@stanford.edu's implementation of the **DeleteOnly** and **DeleteAndRetrieve** models from [Delete, Retrieve, Generate:
A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf). It has also added the following components to the source code:

* optional backtranslation, from [Multiple Attribute Text Rewriting](https://research.fb.com/wp-content/uploads/2019/04/Multiple-Attribute-Text-Rewriting.pdf)
* a *TransformerXL* encoder and decoder with relative position attention
* language model fusion approaches with *GPT* or *GPT2* pre-trained language models, from [Cold Fusion: Training Seq2Seq Models Together with Language Models](https://arxiv.org/pdf/1708.06426.pdf)
* optional adversarial training loop with CNN discriminators, from [Style Transfer from Non-Parallel Text by
Cross-Alignment](https://arxiv.org/pdf/1705.09655.pdf) (primarily) and [Toward Controlled Generation of Text](https://arxiv.org/abs/1703.00955.pdf)
    * specifically, the discriminators differentiate between unrolled hidden states of a teacher-forced sequence to *style_i* and an unforced sequence in *style_i*, generated from a differentiable probability distribution over tokens at each timestep
* supports training on an arbitrary number of styles and inference to multiple styles, from [Multiple Attribute Text Rewriting](https://research.fb.com/wp-content/uploads/2019/04/Multiple-Attribute-Text-Rewriting.pdf)
* supports three noising methods on the input sequences: *random dropout*, *word attribute* selection, and *ngram attribute* selection 
* supports *greedy* and *top-k* decoding
* supports two loss functions: *cross entropy* and a differentiable *lower bound on the expected bleu score*

# Usage

### Training

`train.dockerfile` generates a docker image that can be used for training.

Its default command is `python train.py --config config.json --bleu`, which trains the model using the parameters specificed in `config.json`.

Checkpoints, logs, model outputs, and TensorBoard summaries are written to `/checkpoints/` + `working_dir`, where `working_dir` is specified in the config.

See `config.json` for all of the training options. Important parameters include 

* `model_type` (`delete`, `delete_retrieve`, or `seq2seq`)
* `bt_ratio` (ratio of loss on back-translated samples in the objective function), 
* `discriminator_ratio` (ratio of summed discriminator losses in objective function)
* `tokenizer` (gpt or gpt2, also used as the language model for fusion)
* `encoder` and `decoder` (lstm, transformer)
* `decode` (greedy, top k).

# Questions, feedback, bugs

jeffrey.gleason@newknowledge.io

## Original Developer:

rpryzant@stanford.edu

### Original Acknowledgements

Thanks lots to [Karishma Mandyam](https://github.com/kmandyam) for contributing! 

