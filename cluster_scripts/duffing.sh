#!/bin/bash
python gen_config.py -rp "$PWD" -type free -ext free -dynamics duffing -ndof 2 -noise 0.30 -n_iter 500 -select "partial" \
-l_x 0 -u_x 1 -l_y 0 -u_y 1
python simulation.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_free.ini"
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_free.ini" \
 -e 14 -ne 1 -tr 36 -ph 54 -pl 0 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -ord 2 \
 --headless --comet
