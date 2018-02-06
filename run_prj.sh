#!/usr/bin/env bash
python3 train.py --batch-size=128 --seq-len=100  --lr=0.00003 --expName=improvement \
 --data=data/vctk --outpath=models --act-fcn=relu --fix-model=True --improve-model=True
 #--checkpoint=models/checkpoints/improvement/bestmodel.ckpt
