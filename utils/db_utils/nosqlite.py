#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import json
import pathlib

__all__ = ["connect"]

class NoSQLite(object):
    """NoSQLite"""
    def __init__(self, path2db: str):
        super(NoSQLite, self).__init__()

        self.path2db = pathlib.Path(path2db)
        self.load()


    def load(self):
        if self.path2db.exists():
            self.path2db.resolve()
            with self.path2db.open(mode='r', buffering=-1, encoding="utf-8") as db_f:
                self.db = json.load(db_f)
        else:
            self.db = {}

    def save(self):
        self.path2db.parent.mkdir(parents=True, exist_ok=True)
        with self.path2db.open(mode='w', buffering=-1, encoding="utf-8") as db_f:
            json.dump(self.db, db_f)

def connect(path2db: str):
    return NoSQLite(path2db)
