#!/bin/sh

python3 -m venv env
source env/bin/activate
# for cpu and nvidia:
pip3 install -r requirements.txt
# for amd gpu:
# pip3 install -r requirements-amd.txt --extra-index-url https://download.pytorch.org/whl/rocm5.2
ipython kernel install --user --name=dcml-harmony-and-ornamentation
