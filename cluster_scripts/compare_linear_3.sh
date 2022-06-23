#!/bin/bash
# Generate configuration and data
python gen_config.py -rp "$PWD" -type free -dynamics linear -ndof 3 -noise 0.30 -n_iter 500 -l_x 0.000001 -l_y 0.000001 -u_x 1.0 -u_y 1.0
python simulation.py --root-path "$PWD" --config-path "config/3springmass_linear_free_free.ini"

# train and test RNN encoder
python train.py --root-path "$PWD" --config-path "config/3springmass_linear_free_free.ini" \
 -e 29 -ne 1 -tr 24 -ph 11 -pl 4 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -tenc "rnn" \
 --headless --comet

# train and test BiRNN encoder
python train.py --root-path "$PWD" --config-path "config/3springmass_linear_free_free.ini" \
 -e 29 -ne 1 -tr 24 -ph 11 -pl 4 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -tenc "birnn" \
 --headless --comet

# train and test NODE encoder
python train.py --root-path "$PWD" --config-path "config/3springmass_linear_free_free.ini" \
 -e 29 -ne 1 -tr 24 -ph 11 -pl 4 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -tenc "node" \
 --headless --comet

# train and test Symplectic NODE
python train.py --root-path "$PWD" --config-path "config/3springmass_linear_free_free.ini" \
 -e 29 -ne 1 -tr 24 -ph 11 -pl 4 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 --headless --comet