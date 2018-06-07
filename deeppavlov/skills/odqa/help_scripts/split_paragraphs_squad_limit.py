import argparse
from typing import List, Dict, Union
import json

from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser()

parser.add_argument("input_path", help="path to an input file with text", type=str)
parser.add_argument("output_path", help="path to an output JSON file", type=str)
parser.add_argument("-l", "--squad-limit", help="a number of tokens in a single chunks", type=int,
                    default=500)
parser.add_argument("-f", "--output-format", help="a format of an output data", default='common',
                    choices={'common', 'ranker'}, type=str)


def tokenize_sentences(fpath: str) -> List[str]:
    """
    Tokenize text at a given path to sentences.
    :param fpath: path to an input file with text
    :return text tokenized by sentences
    """
    with open(fpath) as fin:
        data = fin.read()
        return sent_tokenize(data)


def write_result(fpath: str, all_chunks: List[str]) -> None:
    """
    Write a result of chunking to a file in JSON format.
    :param fpath: path to an output file
    :param all_chunks: a result of chunking
    """
    with open(fpath, 'w') as fout:
        json.dump(all_chunks, fout, ensure_ascii=False)


def make_chunks(sentences, squad_limit) -> List[str]:
    all_chunks = []
    _len = 0
    chunks = []
    for sentence in sentences:
        _len += len(sentence.split())
        if _len > squad_limit:
            all_chunks.append(' '.join(chunks))
            _len = 0
            chunks = []
        chunks.append(sentence)

    if chunks:
        all_chunks.append(' '.join(chunks))

    return all_chunks


def make_chunks_2ranker(sentences, squad_limit) -> List[Dict[str, Union[str, int]]]:
    all_chunks = []
    _len = 0
    chunks = []
    i = 0
    for sentence in sentences:
        _len += len(sentence.split())
        if _len > squad_limit:
            d = {'text': ' '.join(chunks), 'title': i}
            all_chunks.append(d)
            _len = 0
            chunks = []
            i += 1
        chunks.append(sentence)

    if chunks:
        d = {'text': ' '.join(chunks), 'title': i}
        all_chunks.append(d)

    return all_chunks


def main():
    args = parser.parse_args()
    sentences = tokenize_sentences(args.input_path)
    f = args.output_format
    limit = args.squad_limit
    if f == 'common':
        all_chunks = make_chunks(sentences, limit)
    elif f == 'ranker':
        all_chunks = make_chunks_2ranker(sentences, limit)
    else:
        raise RuntimeError("Unsupported output format! Select 'common' or 'ranker'.")
    write_result(args.output_path, all_chunks)
    print('Done!')


if __name__ == "__main__":
    main()
