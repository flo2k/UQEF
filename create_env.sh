#!/bin/bash

#create env
python3 -m venv /data/repos/repos_tum/python_venv/uqef_master/

#activate env
. /data/repos/repos_tum/python_venv/uqef_master/bin/activate

#leave env
deactivate




#pyenv: install a specific python version
#https://github.com/pyenv/pyenv

#install pyenv dependencies
sudo apt update
sudo apt install -y --no-install-recommends \
  build-essential make gcc g++ \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  libgdbm-dev libnss3-dev uuid-dev ca-certificates curl git

#install pyenv
curl -fsSL https://pyenv.run | bash

#load pyenv
# Load pyenv automatically by appending
# the following to 
# ~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
# and ~/.bashrc (for interactive shells) :
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"


#install python version
pyenv install 3.11.11
pyenv global 3.11.11


#create env
/home/flo/.pyenv/shims/python3.11 -m venv /data/repos/repos_tum/python_venv/uqef_master_python3_11/

#activate env
. /data/repos/repos_tum/python_venv/uqef_master_python3_11/bin/activate

#leave env
deactivate



############ setup test!

deactivate; rm -rf /data/repos/repos_tum/python_venv/uqef_setup_test/
/home/flo/.pyenv/shims/python3.11 -m venv /data/repos/repos_tum/python_venv/uqef_setup_test/
. /data/repos/repos_tum/python_venv/uqef_setup_test/bin/activate


############ pypi upload
#configure ~/..pypirc with the credentials for pypi and testpypi repos
 python -m pip install --upgrade build twine
 python3 -m build
 python3 -m twine upload --repository testpypi dist/*
 python3 -m twine upload dist/*


#https://pypi.org/project/uqef/
#https://pypi.org/project/uqef/