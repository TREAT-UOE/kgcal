# /bin/bash

mkdir .venv
pipenv --python 3.7
pipenv install protobuf==3.20
pipenv install "tensorflow>=1.15.2,<2.0"
pipenv install ampligraph
pipenv install betacal


