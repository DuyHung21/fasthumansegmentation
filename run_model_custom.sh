#!/bin/bash

export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
workon machine_learning

if [ $# -eq 0 ]; then
    python test.py --mode custom --model_file pretrained/custom.pth        
elif [ $# -eq 1 ]; then
    python test.py --mode custom --model_file pretrained/custom.pth --background "$1"
else 
    python test.py --mode custom --model_file pretrained/custom.pth --background "$1" --vid abc
fi