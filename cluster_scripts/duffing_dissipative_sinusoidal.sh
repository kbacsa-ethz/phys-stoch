#!/bin/bash
python gen_config.py -rp "$PWD" -type dissipative -dynamics duffing -ext sinusoidal -amplitude 1.0 -ndof 2 -noise 0.30 -n_iter 500
python simulation.py --root-path "$PWD" --config-path "config/2springmass_duffing_sinusoidal_dissipative.ini"
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_sinusoidal_dissipative.ini" \
 -e 14 -ne 1 -tr 36 -ph 54 -pl 0 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 --dissipative \
 --headless --comet
