# Pretrained Vietnamese BERT models

This is a repository of pretrained Vietnamese BERT models.
The pretrained models are available along with the source code of pretraining.

## Features

- All the models are trained on [Vietnamese Wikipedia](http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/viwiki/20200520/viwiki-20200520-pages-articles-multistream.xml.bz2).
    
- All the models are trained with the same configuration as the original BERT; 512 tokens per instance, 256 instances per batch, and 1M training steps.
- We also distribute models trained with **Whole Word Masking** enabled; all of the tokens corresponding to a word (tokenized by Underthesea) are masked at once.
- Along with the models, we provide [tokenizers](tokenization.py), which are compatible with ones defined in [Transformers](https://github.com/huggingface/transformers) by Hugging Face.
- using underthesea library to to sentence processing.

- v2 [Update VietNameseTokenierNormalize](https://github.com/manhlab/VietnameseTextNormalizer)
## Pretrained models

- BERT-base models (12-layer, 768-hidden, 12-heads, 110M parameters)
    - **[`BERT-base-bpe-32k.tar.xz`](www.driver.com)** (2.1GB)
        - underthenlpsea + WordPiece tokenization.

All the model archives include following files.
`pytorch_model.bin` and `tf_model.h5` are compatible with [Transformers](https://github.com/huggingface/transformers).

```
.
├── config.json
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
├── pytorch_model.bin
├── tf_model.h5
└── vocab.txt
```

At present, only `BERT-base` models are available.
I am planning to release `BERT-large` models in the future.

## Requirements

For just using the models:

- [Transformers](https://github.com/huggingface/transformers) (== 2.2.2)

If you wish to pretrain a model:

- [TensorFlow](https://github.com/tensorflow/tensorflow) (== 1.14.0)
- [SentencePiece](https://github.com/google/sentencepiece)
- [logzero](https://github.com/metachris/logzero)

## Usage

Please refer to [`masked_lm_example.ipynb`](masked_lm_example.ipynb).

## Details of pretraining

### Corpus generation and preprocessing

The all distributed models are pretrained on Vietnamese Wikipedia.
To generate the corpus, [WikiExtractor](https://github.com/attardi/wikiextractor) is used to extract plain texts from a Wikipedia dump file.

```
$ wget http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/viwiki/20200520/viwiki-20200520-pages-articles-multistream.xml.bz2
$ python wikiextractor/WikiExtractor.py --output /corpus --bytes 512M --compress --json --links --namespaces 0 --no_templates --min_text_length 16 --processes 20 viwiki-20200520-pages-articles-multistream.xml.bz2

install requirements library
$ sudo bash preprocessing.sh
Some preprocessing is applied to the extracted texts.
Preprocessing includes splitting texts into sentences, removing noisy markups, etc.

```sh
$ seq -f %02g 0 3|xargs -L 1 -I {} -P 9 python bert-vietnamese/make_corpus.py --input_file /corpus/AA/wiki_{}.bz2 --output_file /corpus/corpus.txt.{} --vina_dict_path /path/to/neologd/dict/dir/
```

### Building vocabulary

Same as the original BERT, we used byte-pair-encoding (BPE) to obtain subwords.
We used a implementaion of BPE in [SentencePiece](https://github.com/google/sentencepiece).

```sh
# For vocab models
$ !python bert-vietnamese/build_vocab.py --input_file "/corpus/corpus.txt.*" --output_file "/base/vocab.txt" --subword_type bpe --vocab_size 32000

```

### Creating data for pretraining

With the vocabulary and text files above, we create dataset files for pretraining.
Note that this process is highly memory-consuming and takes many hours.

```sh
# For 32k w/ whole word masking
# Note: each process will consume about 32GB RAM
$ seq -f %02g 0 8|xargs -L 1 -I {} -P 1 python create_pretraining_data.py --input_file /path/to/corpus/dir/corpus.txt.{} --output_file /path/to/base/dir/pretraining-data.tf_record.{} --do_whole_word_mask True --vocab_file /path/to/base/dir/vocab.txt --subword_type bpe --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15

# Note: each process will consume about 32GB RAM
$ !seq -f %02g 0 8|xargs -L 1 -I {} -P 1 python bert-vietnamese/create_pretraining_data.py --input_file /corpus/corpus.txt.{} --output_file /base/pretraining-data.tf_record.{} --do_whole_word_mask True --vocab_file /base/vocab.txt --subword_type bpe --max_seq_length 512 --max_predictions_per_seq 80 --masked_lm_prob 0.15

```

### Training

We used [Cloud TPUs](https://cloud.google.com/tpu/) to run pre-training.

For BERT-base models, v3-8 TPUs are used.

```sh
# For BERT-base models
$ python3 run_pretraining.py \
--input_file="/path/to/pretraining-data.tf_record.*" \
--output_dir="/path/to/output_dir" \
--bert_config_file=bert_base_32k_config.json \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--do_train=True \
--train_batch_size=256 \
--num_train_steps=1000000 \
--learning_rate=1e-4 \
--save_checkpoints_steps=100000 \
--keep_checkpoint_max=10 \
--use_tpu=True \
--tpu_name=<tpu name> \
--num_tpu_cores=8

```

## Using

- Model can use with transformers:
    - Tokenizer:
    - Model:
## Licenses

The pretrained models are distributed under the terms of the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

The codes in this repository are distributed under the MIT License.

## Related Work

- Original BERT model by Google Research Team
    - https://github.com/google-research/bert
    - https://github.com/tensorflow/models/tree/master/official/nlp/bert (for TensorFlow 2.0)


- Sentencepiece Vietnamese BERT model
    - Author: Tran Duc Manh
    - https://github.com/manhlab


## Acknowledgments

For training models, we used Cloud TPUs provided by [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc/) program.
Thanks for [Japanese BERT](https://github.com/cl-tohoku/bert-japanese) !
