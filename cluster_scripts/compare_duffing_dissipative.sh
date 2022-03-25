#!/bin/bash
# Generate configuration and data
python gen_config.py -rp "$PWD" -type dissipative -dynamics duffing -ndof 2 -noise 0.30 -n_iter 500 -l_x -2.0 -l_y -2.0 -u_x 2.0 -u_y 2.0
python simulation.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_dissipative.ini"

# train and test RNN encoder
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_dissipative.ini" \
 -e 14 -ne 1 -tr 36 -ph 54 -pl 0 -nenc 1 \
 -sq 50 -n 20 -wd 0.01 \
 -tenc "rnn" \
 --headless --comet

# train and test BiRNN encoder
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_dissipative.ini" \
 -e 14 -ne 1 -tr 36 -ph 54 -pl 0 -nenc 1 \
 -sq 50 -n 20 -wd 0.01 \
 -tenc "birnn" \
 --headless --comet

# train and test NODE encoder
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_dissipative.ini" \
 -e 14 -ne 1 -tr 36 -ph 54 -pl 0 -nenc 1 \
 -sq 50 -n 20 -wd 0.01 \
 -tenc "node" \
 --headless --comet

# train and test Symplectic NODE
python train.py --root-path "$PWD" --config-path "config/2springmass_duffing_free_dissipative.ini" \
 -e 14 -ne 1 -tr 36 -ph 54 -pl 0 -nenc 1 \
 -sq 50 -n 20 -wd 0.01 \
 --headless --comet