#!/bin/bash

while true ; do
    RUN_CLM_RSS=`ps ax -o rss,command | grep -v grep | grep -v _accelerate | grep run_clm.py | awk '{print $1}' | xargs echo -n`

    if [[ -n $RUN_CLM_RSS ]] ; then
        if (( $RUN_CLM_RSS > 22000000 )) ; then
            ACCELERATE_PID=`ps ax -o pid,command | grep -v grep | grep _accelerate | awk '{print $1}'`
            kill $ACCELERATE_PID
            sleep 2
            ACCELERATE_ALIVE=`ps ax -o pid | grep $ACCELERATE_PID | wc -l`
            if (( $ACCELERATE_ALIVE > 0 )) ; then
                sleep 10
                RUN_CLM_PID=`ps ax -o pid,command | grep -v grep | grep run_clm.py | awk '{print $1}'`
                kill -9 $ACCELERATE_PID
                kill -9 $RUN_CLM_PID
            fi
        fi
    fi
    sleep 1
done
