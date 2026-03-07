#!/bin/bash

alias exp="python tools/train_net.py -s 0"
alias dior="exp -d DIOR"
alias dota="exp -d DOTA"

dior -c configs/DIOR/19-1/base.yaml
dior -c configs/DIOR/15-5/base.yaml
dior -c configs/DIOR/10-10/base.yaml
dior -c configs/DIOR/5-15/base.yaml

dota -c configs/DOTA/14-1/base.yaml
dota -c configs/DOTA/10-5/base.yaml
dota -c configs/DOTA/8-7/base.yaml
dota -c configs/DOTA/5-10/base.yaml