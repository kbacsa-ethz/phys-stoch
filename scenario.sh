#!/bin/bash

# Linear
sh pipeline.sh 2 linear free free 20 10 0,1,4,5 0 1 0 1 springmass 18 2 15 85 2 1
sh pipeline.sh 2 linear free free 20 10 0,4  0 1 0 1 springmass 18 2 15 85 2 1
sh pipeline.sh 2 linear free free 20 10 4,5  0 1 0 1 springmass 18 2 15 85 2 1
sh pipeline.sh 2 linear dissipative free 20 10 0,1,4,5 0 1 0 1 springmass 18 2 15 85 2 1
sh pipeline.sh 2 linear dissipative sinusoidal 20 10 0,1,4,5 0 1 0 1 springmass 18 2 15 85 2 1

# Linear 3
sh pipeline.sh 3 linear free free 20 10 0,1,2,6,7,8 0 1 0 1 springmass 29 1 24 11 4 1
sh pipeline.sh 3 linear free free 20 10 0,6  0 1 0 1 springmass 29 1 24 11 4 1
sh pipeline.sh 3 linear free free 20 10 6,7,8  0 1 0 1 springmass 29 1 24 11 4 1
sh pipeline.sh 3 linear dissipative free 20 10 0,1,2,6,7,8 0 1 0 1 springmass 29 1 24 11 4 1
sh pipeline.sh 3 linear dissipative sinusoidal 20 10 0,1,2,6,7,8 0 1 0 1 springmass 29 1 24 11 4 1

# Duffing
sh pipeline.sh 2 duffing free free 20 10 0,1,4,5 0 1 0 1 springmass 14 1 36 54 0 1
sh pipeline.sh 2 duffing free free 20 10 0,4  0 1 0 1 springmass 14 1 36 54 0 1
sh pipeline.sh 2 duffing free free 20 10 4,5  0 1 0 1 springmass 14 1 36 54 0 1
sh pipeline.sh 2 duffing dissipative free 20 10 0,1,4,5 0 1 0 1 springmass 14 1 36 54 0 1
sh pipeline.sh 2 duffing dissipative sinusoidal 20 10 0,1,4,5 0 1 0 1 springmass 14 1 36 54 0 1

# Pendulum
sh pipeline.sh 2 pendulum free free 20 10 0,1,4,5 0 1 0 1 springmass 13 0 18 21 4 1
sh pipeline.sh 2 pendulum free free 20 10 0,4  0 1 0 1 springmass 13 0 18 21 4 1 
sh pipeline.sh 2 pendulum free free 20 10 4,5  0 1 0 1 springmass 13 0 18 21 4 1
sh pipeline.sh 2 pendulum dissipative free 20 10 0,1,4,5 0 1 0 1 springmass 13 0 18 21 4 1 
sh pipeline.sh 2 pendulum dissipative sinusoidal 20 10 0,1,4,5 0 1 0 1 springmass 13 0 18 21 4 1

