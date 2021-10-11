#!/bin/bash
python gen_config.py -rp "$PWD" -type free -dynamics duffing -ndof 2 -noise 0.20 -n_iter 2000
python simulation.py --root-path "$PWD" --config-path "config/2springmass_duffing_free.ini"
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free.ini" \
 -e 14 -ne 3 -tr 26 -ph 23 -pl 4 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 --headless --comet
