#!/usr/bin/env bash
while true;
do
  if [ -e wiki_script.lock ]; then
    echo "script is already running"
  else
    touch wiki_script.lock
    python get_wiki_page_view.py
  fi
  rm wiki_script.lock
  sleep 2;
done