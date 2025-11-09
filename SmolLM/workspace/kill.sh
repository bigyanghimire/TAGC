#!/bin/bash

ps aux | grep /train.sh | grep -v grep | awk '{ print $2 }' | xargs -r -n 1 kill
ps aux | grep _internal_train.sh | grep -v grep | awk '{ print $2 }' | xargs -r -n 1 kill
ps aux | grep _accelerate.py | grep -v grep | awk '{ print $2 }' | xargs -r -n 1 kill
ps aux | grep _oom_killer.sh | grep -v grep | awk '{ print $2 }' | xargs -r -n 1 kill
