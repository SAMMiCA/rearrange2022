#!/bin/bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential
sudo apt-get install -y git libvulkan1

cd ~
mkdir research && cd research
git clone https://github.com/SAMMiCA/rearrange2022.git
cd rearrange2022

# Configure Conda PATH
echo '''
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('~/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "~/miniconda3/etc/profile.d/conda.sh" ]; then
        . "~/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="~/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
''' >> ~/.profile
source ~/.profile

export MY_ENV_NAME=thor-rearrange2022
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"

conda env create --file environment.yml --name $MY_ENV_NAME
conda activate $MY_ENV_NAME

pip3 install setuptools==65.5.0

conda env update --file environment.yml

pip install transforms3d kornia canonicaljson
pip install git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

python -c "import clip; clip.load('RN50', 'cpu')"
python -c "from ai2thor.controller import Controller; from ai2thor.platform import CloudRendering; from rearrange.constants import THOR_COMMIT_ID; c = Controller(commit_id=THOR_COMMIT_ID, platform=CloudRendering); c.stop()"

echo DONE
