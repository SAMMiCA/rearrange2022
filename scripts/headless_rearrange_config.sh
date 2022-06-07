#!/bin/bash
# Prepare a conda env for ai2thor-rearrange
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

# echo '''
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __miniconda_setup="$('~/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# if [ $? -eq 0 ]; then
#     eval "$__miniconda_setup"
# else
#     __anaconda_setup="$('~/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#     if [ $? -eq 0 ]; then
#         eval "$__anaconda_setup"
#     elif [ -f "~/anaconda3/etc/profile.d/conda.sh" ]; then
#         . "~/anaconda3/etc/profile.d/conda.sh"
#     elif [ -d "~/anaconda3/bin" ]; then
#         export PATH="~/anaconda3/bin:$PATH"
#     elif [ -f "~/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "~/miniconda3/etc/profile.d/conda.sh"
#     elif [ -d "~/miniconda3/bin" ]; then
#         export PATH="~/miniconda3/bin:$PATH"
#     fi
#     unset __anaconda_setup
# fi
# unset __miniconda_setup
# # <<< conda initialize <<<
# ''' >> ~/.profile

source ~/.profile

sudo apt-get install -y git libvulkan1

cd ~
mkdir -p research
cd research
git clone https://github.com/kyh2010/rearrange2022.git
cd rearrange2022

export MY_ENV_NAME=rearrange2022
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"

conda env create --file environment.yml --name $MY_ENV_NAME

conda activate rearrange2022

# Download AI2-THOR binaries
python -c "from ai2thor.controller import Controller; from ai2thor.platform import CloudRendering; from rearrange.constants import THOR_COMMIT_ID; c = Controller(platform=CloudRendering, commit_id=THOR_COMMIT_ID); c.stop()"

echo DONE
