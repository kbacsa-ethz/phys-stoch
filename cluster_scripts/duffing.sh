#!/bin/bash
python gen_config.py -rp "$PWD" -type free -dynamics duffing -ndof 2 -noise 0.05
python simulation.py --root-path "$PWD" --config-path "config/2springmass_duffing_free.ini"
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free.ini" \
 -sq 50 -n 30 -wd 0.01 \
 --headless --comet