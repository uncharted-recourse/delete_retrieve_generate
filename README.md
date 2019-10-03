# Description

This repository implements an auto-encoder approach to style transfer dubbed **Ventriloquist**. The model uses the TransformerXL architecture with relative position attention for the encoder and decoder modules and is fused with OpenAI's GPT language model for regularization. 

The **flask.dockerfile** in the main repository builds a docker image that launches a flask app with a translation-like interface. An example screenshot is shown below:

![api_example](https://github.com/NewKnowledge/delete_retrieve_generate/tree/master/screenshots/api_example.png)

Users can input the number of translations, the target style(s), and the decoding parameters k and temperature and observe how these choices affect the styled tranlsations!

# Extensions from Original Implementation

This repository was originally forked from rpryzant@stanford.edu's implementation of the DeleteOnly and DeleteAndRetrieve models from [Delete, Retrieve, Generate:
A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf). It has also added the following components to the source code:

* optional backtranslation, from https://research.fb.com/wp-content/uploads/2019/04/Multiple-Attribute-Text-Rewriting.pdf
* TransformerXL encoder and decoder with relative position attention
* simple fusion with GPT or GPT2 pre-trained language models
* optional adversarial training loop with CNNs as the discriminators, from https://arxiv.org/pdf/1705.09655.pdf (primarily) and https://arxiv.org/abs/1703.00955
    * specifically, the discriminators differentiate between unrolled hidden states of a teacher-forced sequence to style_{i} and an unforced sequence to style_{i}, generated with a soft probability distribution over tokens
* supports training on an arbitrary number of styles and inference to (optionally) multiple styles, from https://research.fb.com/wp-content/uploads/2019/04/Multiple-Attribute-Text-Rewriting.pdf)
* supports three noising methods on input sequences: random dropout, word attribute selection, and ngram attribute selection 
* supports greedy and top k decoding
* supports two loss functions: cross entropy and a differentiable lower bound on the expected bleu score

# Usage

### Training

`train.dockerfile` generates docker image that can be used for training.

It's default command is `python train.py --config config.json --bleu`, which trains the model using the parameters specificed in `config.json`.

Checkpoints, logs, model outputs, and TensorBoard summaries are written to `/checkpoints/` + `working_dir`, where `working_dir` is specified in the config.

See `config.json` for all of the training options. Important parameters include `model_type` (`delete`, `delete_retrieve`, or `seq2seq` (which is a standard translation-style model), `bt_ratio` (ratio of back-translated samples in the objective function, 0 means no back-translated samples are generated), `discriminator_ratio` (ratio of summed discriminator loss terms in objective function, `tokenizer` (gpt or gpt2, which is also used as the language model for fusion), `encoder/decoder` (lstm, transformer), `decode` (greedy, top k).

# Questions, feedback, bugs

jeffrey.gleason@newknowledge.io

# Original Developer:

rpryzant@stanford.edu

# Original Acknowledgements

Thanks lots to [Karishma Mandyam](https://github.com/kmandyam) for contributing! 

