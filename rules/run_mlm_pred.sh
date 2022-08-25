#!/home/graph/jbonnell/miniconda3/envs/adaptesupar/bin/fish
conda activate /home/graph/jbonnell/miniconda3/envs/adaptesupar
set chunk $argv[1]
python mlm_pred.py $chunk

