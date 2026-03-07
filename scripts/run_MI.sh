#!/bin/bash

alias exp="python tools/train_net.py"
alias dior="exp -d DIOR"
alias dota="exp -d DOTA"

for s in {1..2}; do
    dior -c configs/DIOR/10-5/target.yaml -s $s
    dior -c configs/DIOR/10-5/ft.yaml -s $s
done

for s in {1..3}; do
    dior -c configs/DIOR/5-5/target.yaml -s $s
    dior -c configs/DIOR/5-5/ft.yaml -s $s
done

for s in {1..5}; do
    dior -c configs/DIOR/10-2/target.yaml -s $s
    dior -c configs/DIOR/10-2/ft.yaml -s $s
done

for s in {1..5}; do
    dior -c configs/DIOR/15-1/target.yaml -s $s
    dior -c configs/DIOR/15-1/ft.yaml -s $s
done

for s in {1..2}; do
    dota -c configs/DOTA/10-5/target.yaml -s $s
    dota -c configs/DOTA/10-5/ft.yaml -s $s
done

for s in {1..5}; do
    dota -c configs/DOTA/10-1/target.yaml -s $s
    dota -c configs/DOTA/10-1/ft.yaml -s $s
done