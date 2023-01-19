# Instructions
## Setup
Install python environment: <br>
`` pip install -r requirements.txt``

Initialize submodule and checkout to correct branch: <br>
``git submodule update --init --recursive`` <br>
``cd torchdiffeq`` <br>
``git checkout develop`` <br>

## Create dataset
Create configuration file (check arguments for specifications): <br>
``python gen_config.py`` <br>
Generate dataset (pass configuration file as argument): <br>
``python simulate.py --config-path path_to_configuration``
## Train model
Train model (check arguments for model specifications): <br>
``python train.py --config-path path_to_configuration`` <br>
All experiments will be recorded in the ``experiments`` folder.

