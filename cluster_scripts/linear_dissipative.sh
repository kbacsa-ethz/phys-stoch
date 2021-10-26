#!/bin/bash
python gen_config.py -rp "$PWD" -type free -dynamics linear -type dissipative -ndof 2 -noise 0.20 -n_iter 500
python simulation.py --root-path "$PWD" --config-path "config/2springmass_linear_free.ini"
python train.py --root-path "$PWD" --config-path "config/2springmass_linear_free.ini" \
 --dissipative \
 -e 16 -ne 0 -tr 32 -ph 60 -pl 2 -nenc 2 \
 -sq 50 -n 30 -wd 0.01 \
 --headless --comet