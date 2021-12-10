#!/bin/bash
python gen_config.py -rp "$PWD" -type dissipative -dynamics pendulum -ndof 2 -noise 0.20 -n_iter 500  \
-l_x 0 -u_x 1 -l_y 0 -u_y 1
python simulation.py --root-path "$PWD" --config-path "config/2springmass_pendulum_dissipative.ini"
python train.py --root-path "$PWD" --config-path "config/2springmass_pendulum_dissipative.ini" \
 --dissipative \
 -sq 50 -n 30 -wd 0.01 \
 -e 13 -ne 0 -tr 18 -ph 21 -pl 4 -nenc 1 \
 --headless --comet