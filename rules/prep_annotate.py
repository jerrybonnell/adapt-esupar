from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM, TokenClassificationPipeline, pipeline
from collections import defaultdict
import torch
import numpy as np
import spacy
import pickle
import glob
from tqdm import tqdm
import os
import re
from rich.pretty import pprint
import sys

"""
NOTE
this is a technical script that is not immediately necessary for
generation of a rule set, but is an intermediary step in our
experiments. the purpose of this script is to collect
sentences with masked predictions filled in from the four systems
tested into a single file (sent_unq.txt), which need to be passed
as input to a pretrained LM for annotation (e.g., ESUPAR, GINZA)
to see if improvement in annotation is yielded by the substitution.

each line of this file is a sentence which needs to be submitted to a
pretrained LM for annotation.

we collect and annotate sentences in this manner to allow for
trivial parallelization of the above step using GNU parallel
(see run_sent_chunk.py).
"""

sents = []
all_configs = glob.glob("chunk_q2r_esupar_adapt_norules053122/*.txt") + \
    glob.glob("chunk_q2r_esupar_baseline-bert-base-japanese-upos_norules053122/*.txt") + \
    glob.glob("chunk_q2r_esupar_baseline-notaiyo-yasuoka-char-cvg-upos_norules053122/*.txt") + \
    glob.glob("chunk_q2r_esupar_baseline-bccwj-yasuoka-char-cvg-upos_norules053122/*.txt")
assert len(all_configs) == 200 * 4
for fn in all_configs:
    with open(fn, "r") as f:
        sents.append(f.readlines())
sents = [item for sublist in sents for item in sublist]
sents = [s.strip() for s in sents]
sent_unq = list(set(sents))
print(len(sent_unq))
os.makedirs("chunks_q2r_all_esupar", exist_ok=True)
with open("chunks_q2r_all_esupar/sent_unq.txt", "w") as f:
    f.write("\n".join([s for s in sent_unq]))

line_offset = []
with open("chunks_q2r_all_esupar/sent_unq.txt", "rb") as f:
    offset = 0
    for line in f:
        line_offset.append(offset)
        offset += len(line)
    f.seek(0)
print("ok")
with open("chunks_q2r_all_esupar/sent_unq_index.pkl", "wb") as f:
    pickle.dump(line_offset, f)

print(len(sent_unq))
x = np.arange(len(sent_unq))
chunks = [(l[0], l[-1]) for l in np.array_split(x, 2000)]

# to be used for run_sent_chunk.py
with open("chunks_sent", "w") as f:
    [f.write(f"{l[0]} {l[1]}\n") for l in chunks]
