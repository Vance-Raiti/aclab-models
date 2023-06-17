#!/bin/bash
loadenv
retdir=$(pwd)
cd /projectnb/aclab/datasets/dataloaders/dataloaders/basenji
pip install -e
cd $retdir
