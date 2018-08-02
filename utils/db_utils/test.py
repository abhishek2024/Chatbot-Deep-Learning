import nosqlite as nsl
import pytest
import os

key1=123
key2='qweQWE'
key3=True
key4="йцуЙЦУ"

def test_create_db():
    conn = nsl.connect('file.json')
    conn.db['key1']=key1
    conn.db['key2']=key2
    conn.db['key3']=key3
    conn.db['key4']=key4
    conn.save()

def test_load_db1():
    conn = nsl.connect('file.json')
    assert conn.db['key1']==key1
    assert conn.db['key2']==key2
    assert conn.db['key3']==key3
    assert conn.db['key4']==key4

def test_update_db():
    conn = nsl.connect('file.json')
    conn.db={}
    conn.save()


def test_load_db2():
    conn = nsl.connect('file.json')
    assert conn.db.get('key1') is None

def test_end():
    os.remove('file.json')
