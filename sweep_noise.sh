#!/bin/bash
for i in $(seq 0.0 0.05 0.3)
do
  python gen_config.py -rp "$PWD" -type "free" -ndof 2 -noise "$i"
  python simulation.py --root-path "$PWD"
  python train.py --root-path="$PWD" -in 4 -z 4 -enc 4 -sq 50 -n 100 -wd 0.01 --headless --comet
done