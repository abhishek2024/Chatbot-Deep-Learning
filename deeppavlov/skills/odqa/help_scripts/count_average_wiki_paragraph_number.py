"""
This script counts an average number of Wikipedia paragraphs.
"""
import argparse
import logging
import time

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('average_num_of_paragraphs_en.log')
file.setFormatter(fmt)
logger.addHandler(file)

parser = argparse.ArgumentParser()
parser.add_argument('-db_path', help='local path to a Sqlite database.', type=str,
                    default="/media/olga/Data/projects/iPavlov/DeepPavlov/download/odqa/enwiki.db")


def main():
    args = parser.parse_args()
    iterator = SQLiteDataIterator(download_path=args.db_path)
    db_size = len(iterator.doc_ids)

    start_time = time.time()
    paragrahs_num = 0
    curr_articles_num = 0
    for idx in iterator.doc_ids:
        content = iterator.get_doc_content(idx)
        paragraphs = content.split('\n')
        paragraphs = [p for p in paragraphs if p]
        curr_articles_num += 1
        paragrahs_num += len(paragraphs)
        average_par_size = paragrahs_num / curr_articles_num
        logger.info("Current average paragraph num: {}".format(average_par_size))
        logger.info("Processed {} articles.".format(curr_articles_num))
    logger.info("Average paragrahs num per article: {}".format(paragrahs_num / db_size))
    logger.info("Completed in {} seconds.".format(time.time() - start_time))


if __name__ == "__main__":
    main()









