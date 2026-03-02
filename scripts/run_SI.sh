#!/bin/bash

alias exp="python tools/train_net.py"

# -------------- DIOR --------------
exp -c configs/DIOR/19-1/target.yaml
exp -c configs/DIOR/19-1/ft.yaml

exp -c configs/DIOR/15-5/target.yaml
exp -c configs/DIOR/15-5/ft.yaml

exp -c configs/DIOR/10-10/target.yaml
exp -c configs/DIOR/10-10/ft.yaml

exp -c configs/DIOR/5-15/target.yaml
exp -c configs/DIOR/5-15/ft.yaml

# -------------- DOTA --------------
exp -c configs/DOTA/14-1/target.yaml
exp -c configs/DOTA/14-1/ft.yaml

exp -c configs/DOTA/10-5/target.yaml
exp -c configs/DOTA/10-5/ft.yaml

exp -c configs/DOTA/8-7/target.yaml
exp -c configs/DOTA/8-7/ft.yaml

exp -c configs/DOTA/5-10/target.yaml
exp -c configs/DOTA/5-10/ft.yaml