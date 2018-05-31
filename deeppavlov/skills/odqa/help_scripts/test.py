# from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

DB_PATH = '/media/olga/Data/projects/ODQA/data/PMEF/en/001/drones.db'
# iterator = SQLiteDataIterator(DB_PATH)

import sqlite3


def get_doc_ids(conn, name):
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM {}'.format(name))
    ids = [ids[0] for ids in cursor.fetchall()]
    cursor.close()
    return ids

def get_db_name(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    assert cursor.arraysize == 1
    name = cursor.fetchone()[0]
    cursor.close()
    return name

def get_doc_content(conn, name, doc_id):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT text FROM {} WHERE id = ?".format(name),
        (doc_id,)
    )
    result = cursor.fetchone()
    cursor.close()
    return result if result is None else result[0]


conn = sqlite3.connect(DB_PATH, check_same_thread=False)
name = get_db_name(conn)
print(name)
ids = get_doc_ids(conn, name)
print(ids)
print(len(ids))
contents = [get_doc_content(conn, name, id) for id in ids]
print(contents)