#!/bin/bash

alias exp="python tools/train_net.py"
alias dior="exp -d DIOR"
alias dota="exp -d DOTA"

# -------------- DIOR --------------
dior -c configs/DIOR/19-1/target.yaml
dior -c configs/DIOR/19-1/ft.yaml

dior -c configs/DIOR/15-5/target.yaml
dior -c configs/DIOR/15-5/ft.yaml

dior -c configs/DIOR/10-10/target.yaml
dior -c configs/DIOR/10-10/ft.yaml

dior -c configs/DIOR/5-15/target.yaml
dior -c configs/DIOR/5-15/ft.yaml

# -------------- DOTA --------------
dota -c configs/DOTA/14-1/target.yaml
dota -c configs/DOTA/14-1/ft.yaml

dota -c configs/DOTA/10-5/target.yaml
dota -c configs/DOTA/10-5/ft.yaml

dota -c configs/DOTA/8-7/target.yaml
dota -c configs/DOTA/8-7/ft.yaml

dota -c configs/DOTA/5-10/target.yaml
dota -c configs/DOTA/5-10/ft.yaml