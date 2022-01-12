#!/bin/bash
# Generate configuration and data
python gen_config.py -rp "$PWD" -type free -dynamics linear -ndof 2 -noise 0.20 -n_iter 500
python simulation.py --root-path "$PWD" --config-path "config/2springmass_linear_free_free.ini"

# train and test RNN encoder
python train.py --root-path "$PWD" --config-path "config/2springmass_linear_free_free.ini" \
 -e 18 -ne 2 -tr 15 -ph 85 -pl 2 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -tenc "rnn" \
 --headless --comet

# train and test BiRNN encoder
python train.py --root-path "$PWD" --config-path "config/2springmass_linear_free_free.ini" \
 -e 18 -ne 2 -tr 15 -ph 85 -pl 2 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -tenc "birnn" \
 --headless --comet

# train and test NODE encoder
python train.py --root-path "$PWD" --config-path "config/2springmass_linear_free_free.ini" \
 -e 18 -ne 2 -tr 15 -ph 85 -pl 2 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 -tenc "node" \
 --headless --comet

# train and test Symplectic NODE
python train.py --root-path "$PWD" --config-path "config/2springmass_linear_free_free.ini" \
 -e 18 -ne 2 -tr 15 -ph 85 -pl 2 -nenc 1 \
 -sq 50 -n 30 -wd 0.01 \
 --headless --comet