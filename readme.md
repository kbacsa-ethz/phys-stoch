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

## Citation 
If you use our code, please kindly cite our work: 
```bibtex
@ARTICLE{Bacsa:2023,
    TITLE        = "Symplectic Encoders for Physics-Constrained Variational Dynamics Inference",
    JOURNAL      = "Scientific Reports",
    VOLUME       = 13,
    NUMBER       = 1,
    PAGES        = 2643,
    YEAR         = 2023,
    MONTH        = "Feb",
    DAY          = 14,
    DOI          = "https://doi.org/10.1038/s41598-023-29186-8",
    AUTHOR       = "Bacsa, Kiran and Lai, Zhilu and Liu, Wei and Todd, Michael and Chatzi, Eleni",
    ABSTRACT     = "We propose a new variational autoencoder (VAE) with physical constraints capable of learning the dynamics of Multiple Degree of Freedom (MDOF) dynamic systems. Standard variational autoencoders place greater emphasis on compression than interpretability regarding the learned latent space. We propose a new type of encoder, based on the recently developed Hamiltonian Neural Networks, to impose symplectic constraints on the inferred a posteriori distribution. In addition to delivering robust trajectory predictions under noisy conditions, our model is capable of learning an energy-preserving latent representation of the system. This offers new perspectives for the application of physics-informed neural networks on engineering problems linked to dynamics.",
    ISSN         = "2045-2322",
    URL          = "https://doi.org/10.1038/s41598-023-29186-8",
    ID           = "Bacsa2023",
}
        }
```
