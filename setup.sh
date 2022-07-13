# /bin/bash
conda create --name ampligraph python=3.7
source activate ampligraph

conda install tensorflow'>=1.15.2,<2.0.0'

pip install ampligraph
pip install betacal
pip install jupyter
