"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
import datetime
from datetime import timedelta as dt

DAYS = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}


log = get_logger(__name__)


@register('time_parser')
class TimeParser(Component):
    def __init__(self, **kwargs):
        pass

    def _parse_time(self, time_tags_batch):
        today = datetime.date.today()
        times = []
        log.debug("time_tags_batch = {}".format(time_tags_batch))
        for time_tag_list in time_tags_batch:
            if len(time_tag_list) > 0:
                tag, score = max(time_tag_list.items(), key=lambda x: x[1])
                if tag == 'today':
                    start = stop = today
                elif tag == 'tomorrow':
                    start = stop = today + dt(days=1)
                elif tag == 'week':
                    start = today
                    stop = today + dt(days=6 - today.weekday())
                elif tag == 'next_week':
                    start = today + dt(days=7 - today.weekday())
                    stop = start + dt(days=6)
                elif tag == 'this_month':
                    start = today
                    stop = self.last_day_of_month(today)
                elif tag == 'following_month':
                    start = today
                    stop = today + dt(days=30)
                elif tag == 'next_month':
                    start = today.replace(month=today.month + 1, day=1)
                    stop = self.last_day_of_month(start)
                elif tag == 'the_day_after_tomorrow':
                    start = stop = today + dt(days=2)
                else:
                    weekday = DAYS[tag]
                    if weekday >= today.weekday():
                        start = stop = today + dt(days=weekday - today.weekday())
                    else:
                        start = stop = today + dt(days=7 - today.weekday() + weekday)
                stop = stop + dt(days=1)
                times.append([start, stop])
            else:
                times.append(None)
        return times

    def __call__(self, time_tags_batch, slot_history):
        times = self._parse_time(time_tags_batch)
        for time, slot_h in zip(times, slot_history):
            slot_h['time_span'] = time or slot_h.get('time_span')
        return slot_history

    @staticmethod
    def last_day_of_month(any_day):
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
        return next_month - datetime.timedelta(days=next_month.day)
