# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team,
# and Masatoshi Suzuki.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Vietnamese BERT models."""

import collections
import logging
import os
import unicodedata
from underthesea import sent_tokenize

from transformers import BertTokenizer, WordpieceTokenizer
from transformers.tokenization_bert import load_vocab


logger = logging.getLogger(__name__)


class VinaBertTokenizer(BertTokenizer):
    """BERT tokenizer for Vietnamese text; underthesea tokenization + WordPiece"""

    def __init__(self, vocab_file, do_lower_case=False,
                 do_basic_tokenize=True, do_wordpiece_tokenize=True,
                 vina_dict_path=None, unk_token='[UNK]', sep_token='[SEP]',
                 pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', **kwargs):
        """Constructs a underthesea BertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
                Only has an effect when do_basic_tokenize=True.
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization with underthesea before wordpiece.
            **vina_dict_path**: (`optional`) string
                Path to a directory of a underthesea dictionary.
        """
        super(BertTokenizer, self).__init__(
            unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
            cls_token=cls_token, mask_token=mask_token, **kwargs)

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        self.do_wordpiece_tokenize = do_wordpiece_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = VinaBasicTokenizer(do_lower_case=do_lower_case,
                                                       vina_dict_path=vina_dict_path)

        if do_wordpiece_tokenize:
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                          unk_token=self.unk_token)

    def _tokenize(self, text):
        if self.do_basic_tokenize:
            tokens = self.basic_tokenizer.tokenize(text,
                                                   never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_wordpiece_tokenize:
            split_tokens = [sub_token for token in tokens
                            for sub_token in self.wordpiece_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens


class VinaCharacterBertTokenizer(BertTokenizer):
    """BERT character tokenizer for with information of Vi tokenization"""

    def __init__(self, vocab_file, do_lower_case=False, do_basic_tokenize=True,
                 vina_dict_path=None, unk_token='[UNK]', sep_token='[SEP]',
                 pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', **kwargs):
        """Constructs a VinaCharacterBertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
                Only has an effect when do_basic_tokenize=True.
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization with Vi before wordpiece.
            **vina_dict_path**: (`optional`) string
                Path to a directory of a Vi dictionary.
        """
        super(BertTokenizer, self).__init__(
            unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
            cls_token=cls_token, mask_token=mask_token, **kwargs)

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = VinaBasicTokenizer(do_lower_case=do_lower_case,
                                                       vina_dict_path=vina_dict_path,
                                                       preserve_spaces=True)

        self.wordpiece_tokenizer = VinaCharacterTokenizer(vocab=self.vocab,
                                                      unk_token=self.unk_token,
                                                      with_markers=True)

    def _convert_token_to_id(self, token):
        """Converts a token (str/unicode) to an id using the vocab."""
        if token[:2] == '##':
            token = token[2:]

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) to a single string."""
        out_string = ' '.join(tokens).replace('##', '').strip()
        return out_string


class VinaBasicTokenizer(object):

    def __init__(self, do_lower_case=False, never_split=None,
                 vina_dict_path=None, preserve_spaces=False):
        """Constructs a VinaBasicTokenizer.

        Args:
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
            **vina_dict_path**: (`optional`) string
                Path to a directory of a Vi dictionary.
            **preserve_spaces**: (`optional`) boolean (default True)
                Whether to preserve whitespaces in the output tokens.
        """
        if never_split is None:
            never_split = []

        self.do_lower_case = do_lower_case
        self.never_split = never_split


        self.preserve_spaces = preserve_spaces

    def tokenize(self, text, never_split=None, with_info=False, **kwargs):
        """Tokenizes a piece of text."""
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = unicodedata.normalize('NFKC', text)

        tokens = []
        token_infos = []

        cursor = 0
        for line in sent_tokenize(text):
            if line == 'EOS':
                if self.preserve_spaces and len(text[cursor:]) > 0:
                    tokens.append(text[cursor:])
                    token_infos.append(None)

                break
            token = line
            token_start = text.index(token, cursor)
            token_end = token_start + len(token)
            if self.preserve_spaces and cursor < token_start:
                tokens.append(text[cursor:token_start])

            if self.do_lower_case and token not in never_split:
                token = token.lower()

            tokens.append(token)

            cursor = token_end

        return tokens


class VinaCharacterTokenizer(object):
    """Runs Character tokenziation."""

    def __init__(self, vocab, unk_token,
                 max_input_chars_per_word=100, with_markers=True):
        """Constructs a VinaCharacterTokenizer.
        Args:
            vocab: Vocabulary object.
            unk_token: A special symbol for out-of-vocabulary token.
            with_markers: If True, "#" is appended to each output character except the
                first one.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.with_markers = with_markers

    def tokenize(self, text):
        """Tokenizes a piece of text into characters.

        For example:
            input = "apple"
            output = ["a", "##p", "##p", "##l", "##e"]  (if self.with_markers is True)
            output = ["a", "p", "p", "l", "e"]          (if self.with_markers is False)
        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.
        Returns:
            A list of characters.
        """

        output_tokens = []
        for i, char in enumerate(text):
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            if self.with_markers and i != 0:
                output_tokens.append('##' + char)
            else:
                output_tokens.append(char)

        return output_tokens
