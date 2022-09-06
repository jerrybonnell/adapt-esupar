#!/home/graph/jbonnell/miniconda3/envs/blueberry/bin/fish
conda activate /home/graph/jbonnell/miniconda3/envs/electra
set -x CUDA_VISIBLE_DEVICES ""
set chunk $argv[1]
python run_sent_chunk.py $chunk

