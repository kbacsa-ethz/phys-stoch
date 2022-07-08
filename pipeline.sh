#!/bin/bash

echo ""
var=$(date)
var=`date`
echo "Experiment at $var"

echo ""
echo "----------PARSING DATA ARGUMENTS----------"
# Parse arguments
ndof="$1"
dynamics="$2"
disp="$3"
ext="$4"
noise="$5"
niter="$6"
select="$7"
lx="$8"
ux="$9"
ly="${10}"
uy="${11}"
sys="${12}"

echo "The unit system is a $sys."
echo "Number of DOFs is $ndof."
echo "System dynamics is $dynamics."
echo "Dissipation-wise, the system is $disp."
echo "Force-wise, the system is $ext."
echo "Base noise is $noise dB."
echo "Simulation iterations is $niter."
echo "The selected obsertations are $select."
echo "The lower bound on the first DOF is $lx."
echo "The upper bound on the first DOF is $ux."
echo "The lower bound on the second DOF is $ly."
echo "The upper bound on the second DOF is $uy."

echo ""
echo "----------PARSING MODEL ARGUMENTS----------"
emi="${13}"
ne="${14}"
trans="${15}"
poth="${16}"
potl="${17}"
nenc="${18}"

echo "Emission dimension of DMM is $emi."
echo "Number of emission layers is $ne."
echo "Transmission dimension of DMM is $trans."
echo "Potential dimension of DMM is $poth."
echo "Number of potential layers is $potl."
echo "Number of encoder layers is $nenc."

echo ""
echo "----------GENERATING CONFIGURATION----------"
python gen_config.py -rp "$PWD" -type $disp -ext $ext -dynamics $dynamics -ndof $ndof -noise $noise -n_iter $niter -select $select \
	-l_x $lx -u_x $ux -l_y $ly -u_y $uy 
echo "Configuration generated"

echo ""
echo "----------GENERATING DATASET----------"
echo "Simulating config/${ndof}_${sys}_${dynamics}_${ext}_${disp}.ini"
python simulation.py --root-path "$PWD" --config-path "config/${ndof}_${sys}_${dynamics}_${ext}_${disp}.ini"

echo ""
echo "----------TRAINING----------"
python train.py --root-path "$PWD" --config-path "config/${ndof}_${sys}_${dynamics}_${ext}_${disp}.ini" \
 -e $emi -ne $ne -tr $trans -ph $poth -pl $potl -nenc $nenc \
 --dissipative $disp \
 -sq 50 -n 1 -wd 0.01 \
 -ord 2 \


echo ""
echo "----------TESTING----------"
python test_model.py --root-path "$PWD" --ckpt-path "${ndof}_${sys}_${dynamics}_${ext}_${disp}"

