# Setup
## Create working directory:
>sudo mkdir /app
>sudo chmod 777 /app

## Install dependencies
pip install -r requirements.txt

pip install diffusers["torch"] transformers

## GIT
git init
git config --global --add safe.directory /app
git remote add origin https://github.com/sbruksha/testOCR.git
git branch -M main
git push -u origin main

conda install gh --channel conda-forge	