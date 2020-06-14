#!/bin/bash
file="./logs/gunicorn.pid" #the file where you keep your string name
pid=$(cat "$file") #the
kill $pid
#kill -TERM 'cat ./logs/gunicorn.pid'
