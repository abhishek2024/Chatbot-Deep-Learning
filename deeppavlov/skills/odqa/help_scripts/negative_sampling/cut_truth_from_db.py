import argparse
import sqlite3
import re
import unicodedata

from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

parser = argparse.ArgumentParser()

parser.add_argument("-input_db_path",
                    help="path to an input db with wikipedia articles \
                    (BETTER MAKE A COPY SINCE IT WILL BE CHANGED IN RUNTIME)",
                    type=str,
                    default="/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_cut_truth.db")
parser.add_argument("-dataset_path", help="path to a SQuAD dataset preprocessed for ranker", type=str,
                    default="/media/olga/Data/datasets/squad/preproc/train-v1.1_prep_4ranker.json")
parser.add_argument("-squad_articles", help="path to a JSON with SQuAD articles", type=str,
                    default="/media/olga/Data/datasets/squad/wiki_articles/squad_train_articles.json")
parser.add_argument("-relevant_titles_path", help="path to a JSON with SQuAD articles", type=str,
                    default="/media/olga/Data/projects/DeepPavlov/deeppavlov/skills/odqa/help_scripts/relevant_ids.json")

NEW2OLD_WIKI_TITLES_TRAIN = {  # new titles are keys, old are values
        'Sky UK': 'Sky (United Kingdom)',
        'Huguenots': 'Huguenot',
        'Phonograph record': 'Gramophone record',
        'Sony Music': 'Sony Music Entertainment',
        'Mary, mother of Jesus': 'Mary (mother of Jesus)',
        'Multiracial Americans': 'Multiracial American',
        'Endangered Species Act of 1973': 'Endangered Species Act',
        'Cardinal (Catholic Church)': 'Cardinal (Catholicism)',
        'Universal Pictures': 'Universal Studios',
        # 'Bill and Melinda Gates Foundation': 'Bill & Melinda Gates Foundation',
        'Gates Foundation': 'Bill & Melinda Gates Foundation',
        'Videotelephony': 'Videoconferencing',
        'Imamah (Shia)': 'Imamah (Shia doctrine)',
        'Seven Years\' War': 'Seven_Years%27_War',
        'Molotov–Ribbentrop Pact': 'Molotov%E2%80%93Ribbentrop_Pact',
        'Bill & Melinda Gates Foundation': 'Bill_%26_Melinda_Gates_Foundation',
        'Brasília': 'Bras%C3%ADlia',
        'Kievan Rus\'': 'Kievan_Rus%27',
        'St. John\'s, Newfoundland and Labrador': 'St._John%27s,_Newfoundland_and_Labrador',
        'Saint Barthélemy': 'Saint_Barth%C3%A9lemy',
        'Financial crisis of 2007–2008': 'Financial_crisis_of_2007%E2%80%9308',
        'Jehovah\'s Witnesses': 'Jehovah%27s_Witnesses',
        'Beyoncé': 'Beyonc%C3%A9'
    }


def encode_utf8(s: str):
    return unicodedata.normalize('NFD', s)

def cut_context_from_article(article, context):

    replace_p = re.compile('[()\",\'\[\]]')
    format_article = re.sub(replace_p, ' ', article)
    format_context = re.sub(replace_p, ' ', context)

    p1 = re.compile(f'{format_context[:10]}.*\n', flags=re.I & re.X & re.A)
    # p2 = re.compile(f'\n.*{format_context[:-10]}', flags=re.I & re.X & re.A)
    p3 = re.compile(f'\n.*{format_context[20:40]}.*\n', flags=re.I & re.X & re.A)
    p4 = re.compile(f'\n.*{format_context[40:60]}.*\n', flags=re.I & re.X & re.A)
    # p5 = re.compile(f'\n.*{format_context[60:80]}.*\n', flags=re.I & re.X & re.A)
    # p6 = re.compile(f'\n.*{format_context[80:100]}.*\n', flags=re.I & re.X & re.A)
    # p7 = re.compile(f'\n.*{format_context[100:120]}.*\n', flags=re.I & re.X & re.A)
    # p8 = re.compile(f'\n.*{format_context[120:140]}.*\n', flags=re.I & re.X & re.A)
    p9 = re.compile(f'\n.*{format_context[140:160]}.*\n', flags=re.I & re.X & re.A)
    # p10 = re.compile(f'\n.*{format_context[160:180]}.*\n', flags=re.I & re.X & re.A)
    # p11 = re.compile(f'\n.*{format_context[180:200]}.*\n', flags=re.I & re.X & re.A)
    # p12 = re.compile(f'\n.*{format_context[200:220]}.*\n', flags=re.I & re.X & re.A)
    # p13 = re.compile(f'\n.*{format_context[-40:-20]}.*\n', flags=re.I & re.X & re.A)
    # p14 = re.compile(f'\n.*{format_context[-80:-60]}.*\n', flags=re.I & re.X & re.A)

    # for TRAIN set:
    cut_patterns = [p1, p3, p4, p9]

    # for DEV set:
    # cut_patterns = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

    failed_count = 0
    cut_article = format_article

    for p in cut_patterns:
        cut_article = re.sub(p, '', format_article)
        if len(format_article) != len(cut_article):
            break

    if len(format_article) == len(cut_article):
        failed_count += 1

    return cut_article, failed_count


def find_wiki_title(title):
    format_title = title.replace('_', ' ')
    if format_title in NEW2OLD_WIKI_TITLES_TRAIN.values():
        wiki_title = [k for k, v in NEW2OLD_WIKI_TITLES_TRAIN.items() if v == format_title][0]
        return wiki_title
    else:
        return format_title


def main():
    args = parser.parse_args()
    db_path = args.input_db_path
    dataset = read_json(args.dataset_path)
    # dataset = dataset[:10]
    print(f"Dataset size: {len(dataset)}")
    squad_articles = read_json(args.squad_articles)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    iterator = SQLiteDataIterator(db_path)
    failed = 0

    # wiki_titles = set()
    # relevant_titles = wiki_titles.union(set(read_json(args.relevant_titles_path)))

    try:
        for i, instance in enumerate(dataset):
            print(f"Processing instance {i}")
            # question = instance['question']
            context = instance['context']
            title = instance['title']

            wiki_title = find_wiki_title(title)
            new_title = ''

            if wiki_title not in iterator.doc_ids:
                if wiki_title in NEW2OLD_WIKI_TITLES_TRAIN.keys():
                    new_title = NEW2OLD_WIKI_TITLES_TRAIN[wiki_title]
                    if new_title not in iterator.doc_ids:
                        new_title = encode_utf8(wiki_title)
                        if new_title not in iterator.doc_ids:
                            print(f"Failed to find title: {wiki_title}")

            if new_title:
                wiki_title = new_title

            true_article = [item['text'] for item in squad_articles if item['title'] == title][0]
            cut_article, _failed = cut_context_from_article(true_article, context)
            # if len(cut_article) >= len(true_article):
            #     print("BIGGER")
            # else:
            #     print("CUT IS SMALLER")

            failed += _failed

            # cursor.execute(
            #     "REPLACE INTO documents VALUES (?,?)", (wiki_title, cut_article))

            # wiki_titles.add(wiki_title)

            cursor.execute("UPDATE documents SET text=? WHERE id=?", (cut_article, wiki_title))
            # cursor.execute(
            #     "SELECT text FROM documents WHERE id = ?",
            #     (wiki_title,)
            # )
            # result = cursor.fetchone()[0]
            # if result != cut_article:
            #     print("NO")

            # cursor.execute("DELETE FROM documents WHERE id=?", (wiki_title,))
            # cursor.execute("INSERT INTO documents VALUES (?,?)", (wiki_title, cut_article))
            i += 1

    # cursor.executemany("DELETE FROM documents WHERE id!=?", (*relevant_ids,))

    except Exception:
        raise
    finally:
        print(f'Number of failed to replace: {failed}')  # 2764

    cursor.close()
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
