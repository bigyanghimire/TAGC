#!/bin/bash

git pull

nohup ./_internal_train.sh $* 2>&1 &
disown
