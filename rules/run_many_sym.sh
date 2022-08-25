#!/usr/bin/bash
cat chunks_adapt_norules053122 | parallel --sshloginfile ../Rules2UD/nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_modify.sh"
# cat chunks_esupar_norules053122 | parallel --sshloginfile ../Rules2UD/nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_modify.sh"
cat chunks_baseline-bert-base-japanese-upos_norules053122 | parallel --sshloginfile ../Rules2UD/nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_modify.sh"
cat chunks_baseline-notaiyo-yasuoka-char-cvg-upos_norules053122 | parallel --sshloginfile ../Rules2UD/nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_modify.sh"
cat chunks_baseline-bccwj-yasuoka-char-cvg-upos_norules053122 | parallel --sshloginfile ../Rules2UD/nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_modify.sh"
