#!/usr/bin/env bash

echo off 
cd src
DATA=../data

# check if input args is non-empty; if not empty use it directly; 
# example ARGS is given in comment
if [ $# -gt 0 ]; 
then
    make run ARGS="$@"
    #make run ARGS="-input=../data/sloth.png -maskSize=25 -filter=1"
else
# input ARGS is empty; then just process all files in 'data' folder
    FILES=$DATA/*

    for f in $FILES 
    do
        make run ARGS="-input=$f -maskSize=25 -filter=2"
        #make run ARGS="-input=$DATA/* -maskSize=25 -filter=1"
    done
fi

