#!/bin/bash
python gen_config.py -rp "$PWD" -type free -dynamics linear -ndof 3 -noise 0.30 -n_iter 500 \
-l_x 0 -u_x 1 -l_y 0 -u_y 1
python simulation.py --root-path "$PWD" --config-path "config/3springmass_linear_free.ini"
python train.py --root-path "$PWD" --config-path "config/3springmass_linear_free.ini" \
 -e 29 -ne 1 -tr 24 -ph 11 -pl 4 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -ord 2 \
 --headless --comet
