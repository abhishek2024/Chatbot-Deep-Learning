import argparse
import sqlite3
from typing import List, Union
from itertools import chain

from nltk.tokenize import sent_tokenize

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

parser = argparse.ArgumentParser()

parser.add_argument("-input_db_path",
                    help="path to an input db with wikipedia articles \
                    (BETTER MAKE A COPY SINCE IT WILL BE CHANGED IN RUNTIME)",
                    type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_cut_irrelevant.db")
parser.add_argument("-new_db_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_chunk.db")


def make_chunks(batch_docs: List[Union[str, List[str]]], keep_sentences=True, tokens_limit=500,
                flatten_result=True) -> List[Union[List[str], List[List[str]]]]:
    result = []

    for docs in batch_docs:
        batch_chunks = []
        if isinstance(docs, str):
            docs = [docs]
        for doc in docs:
            doc_chunks = []
            if keep_sentences:
                sentences = sent_tokenize(doc)
                n_tokens = 0
                keep = []
                for s in sentences:
                    n_tokens += len(s.split())
                    if n_tokens > tokens_limit:
                        if keep:
                            doc_chunks.append(' '.join(keep))
                            n_tokens = 0
                            keep.clear()
                    keep.append(s)
                if keep:
                    doc_chunks.append(' '.join(keep))
                batch_chunks.append(doc_chunks)
            else:
                split_doc = doc.split()
                doc_chunks = [split_doc[i:i + tokens_limit] for i in
                              range(0, len(split_doc), tokens_limit)]
                batch_chunks.append(doc_chunks)
        result.append(batch_chunks)

    if flatten_result:
        if isinstance(result[0][0], list):
            for i in range(len(result)):
                result[i] = list(chain.from_iterable(result[i]))

    return result


def main():
    args = parser.parse_args()
    db_path = args.input_db_path
    new_db_path = args.new_db_path

    # conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()

    iterator = SQLiteDataIterator(db_path)

    conn = sqlite3.connect(new_db_path)
    cursor = conn.cursor()
    sql_table = "CREATE TABLE documents (id PRIMARY KEY, text);"
    cursor.execute(sql_table)

    chunk_id = 0

    for i in iterator.doc_ids:
        content = iterator.get_doc_content(i)
        chunks = make_chunks([content])
        for chunk in chain.from_iterable(chunks):
            cursor.execute("INSERT INTO documents VALUES (?,?)", (chunk_id, chunk))
            chunk_id += 1

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
