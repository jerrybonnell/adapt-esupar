#!/home/graph/jbonnell/miniconda3/envs/blueberry/bin/fish
conda activate /home/graph/jbonnell/miniconda3/envs/electra
set chunk $argv[1]
python mlm_pred.py $chunk

