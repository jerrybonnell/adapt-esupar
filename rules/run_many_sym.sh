#!/usr/bin/bash
cat chunks_adapt_norules053122 | parallel --sshloginfile nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/adapt-esupar/rules && ./run_mlm_pred.sh"
# cat chunks_esupar_norules053122 | parallel --sshloginfile nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_mlm_pred.sh"
cat chunks_baseline-bert-base-japanese-upos_norules053122 | parallel --sshloginfile nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/adapt-esupar/rules && ./run_mlm_pred.sh"
cat chunks_baseline-notaiyo-yasuoka-char-cvg-upos_norules053122 | parallel --sshloginfile nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/adapt-esupar/rules && ./run_mlm_pred.sh"
cat chunks_baseline-bccwj-yasuoka-char-cvg-upos_norules053122 | parallel --sshloginfile nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/adapt-esupar/rules && ./run_mlm_pred.sh"
