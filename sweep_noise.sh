#!/bin/bash
for i in $(seq 0.01 0.03 0.06 0.1 0.13 0.16 0.2)
do
  python gen_config.py -rp "$PWD" -type "free" -ndof 2 -noise "$i"
  python simulation.py --root-path "$PWD"
  python train.py --root-path="$PWD" -in 4 -z 4 -enc 4 -sq 50 -n 100 -wd 0.01 --headless --comet
done