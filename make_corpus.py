import re
import bz2
import json
import unicodedata
import argparse
import regex
from logzero import logger
from underthesea import sent_tokenize
from underthesea import word_tokenize

class VinaSentenceSplitter(object):
    def __init__(self):
        self
    def __call__(self, text):
        return sent_tokenize(text)

def preprocess_text(text):
    text = re.sub(r'、+', '', text)
    text = text.replace('(、', '(')
    text = text.replace('、)', ')')
    text = text.replace('()', '')
    text = re.sub(r'\s+', ' ', text)
    text = regex.sub(r'[^\p{Latin}]', u'', text)
    return text.strip()


def filter_text(text, min_length, max_length):
    if re.search(r'\| *\|+', text):
        return False
    if len(text) < min_length or len(text) > max_length:
        return False

    return True

regex_link = re.compile(r'\<a href="(.*?)"\>(.*?)\</a\>')


def main(args):
    sent_splitter = VinaSentenceSplitter()
    
    num_processed_docs = 0
    with bz2.open(args.input_file, 'rt') as input_file, \
         open(args.output_file, 'w') as output_file:
        for line in input_file:
            page_item = json.loads(line)
            text = page_item['text'].lower()

            # replace links
            text = regex_link.sub(r'\2', text)

            # normalize text
            text = unicodedata.normalize('NFKC', text)

            paragraphs = re.split(r'\n\n+', text)[1:]
            sentences = [preprocess_text(s) for p in paragraphs
                         for s in sent_splitter(p)]
            # ignore too short/long sentences
            sentences = [s for s in sentences
                         if filter_text(s, args.min_length, args.max_length)]
            if sentences:
                # write document to a file
                for s in sentences:
                    assert not '\n' in s, s
                    assert s, s
                    output_file.write(s + '\n')

                output_file.write('\n')

            num_processed_docs += 1
            if args.debug and num_processed_docs == 1000:
                logger.info('processed: {}'.format(num_processed_docs))
                break

            # logging
            if num_processed_docs % 10000 == 0:
                logger.info('processed: {}'.format(num_processed_docs))

        if num_processed_docs % 10000 != 0:
            logger.info('processed: {}'.format(num_processed_docs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
        help='preprocessed Wikipedia articles file (.bz2)')
    parser.add_argument('--output_file', type=str, required=True,
        help='output corpus file')
    parser.add_argument('--min_length', type=int, default=16,
        help='only extract sentences with no less than N characters [16]')
    parser.add_argument('--max_length', type=int, default=1024,
        help='only extract sentences with no more than N characters [1024]')
    parser.add_argument('--Vina_dict_path', type=str,
        help='path to Vina dictionary')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
