#!/usr/bin/env bash
echo "Testing TTS"

CHECKPOINT="models/checkpoints/improvement/bestmodel.ckpt"

echo 'python3 generator.py --spkr=0 --text="Hello, this is a test" --checkpoint=${CHECKPOINT}'
python3 generator.py --spkr=0 --text="Hello, this is a test" --checkpoint=${CHECKPOINT}
python generate_merlin_data.py

echo 'python3 generator.py --spkr=0 --npz=data/vctk/numpy_features_valid/p318_212.npz --checkpoint=${CHECKPOINT}'
python3 generator.py --spkr=0 --npz=data/vctk/numpy_features_valid/p318_212.npz --checkpoint=${CHECKPOINT}
python generate_merlin_data.py

echo "Done!"
