from collections import defaultdict
import numpy as np
import spacy
import pickle
import sys
import glob
import re
from tqdm import tqdm
import os
from transformers import pipeline
import torch
from rich.pretty import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM, TokenClassificationPipeline, pipeline

print(sys.argv)
assert len(sys.argv) == 2
begin_chunk, end_chunk, op, experiment, m_path = sys.argv[1].split(" ")
begin_chunk = int(begin_chunk)
end_chunk = int(end_chunk)
op = int(op)
print((begin_chunk, end_chunk, op, experiment, m_path))

if op == 0:
    """
    for speed-up this should be run in a parallelization mode, e.g.,
    cat chunks | parallel --sshloginfile ../Rules2UD/nodeslist_new --jobs 2 "cd /home/lab/jbonnell/taiyo/symbolism && ./run_mlm_pred.sh"
    """
    nlp = pipeline("fill-mask", model=m_path, top_k=15)

    with open(f"query2context_esupar_{experiment}.pkl", "rb") as f:
        query2context = pickle.load(f)
    # NOTE work on a portion of query2context to parallelize the work using GNU parallel
    query2context = dict(sorted(query2context.items())[begin_chunk:end_chunk+1])
    query2sents = {}
    for query, contexts in tqdm(query2context.items()):
        # key: rule pairs predicted by the MLM of BERT
        # value: list of tuples containing context-test sentence pairs
        #        for this rule pair
        # the MLM may predict some given rule pair from multiple different contexts
        # this should be at most len(contexts), the total number of contexts
        cand_ginza = defaultdict(list)
        for context, masked_contexts, _ in tqdm(contexts):
            # [('兄貴の臑ツ嚼だの、', [('X嚼', '兄貴の臑[MASK]嚼だの、'), ('ツX', '兄貴の臑ツ[MASK]だの、')])]
            assert len(masked_contexts) % 2 == 0
            tups = []
            for rule, masked_context in masked_contexts:
                for tok in [res['token_str'] for res in nlp(masked_context)]:
                    tups.append((query, tok, rule.replace("X", tok), masked_context))
            assert len(tups) == len(masked_contexts) * 15 # 30 if simple case

            for query, mask_pred, rule, masked_context in tups:
                # test_sent = context.replace(query, rule, 1)
                assert masked_context.count(nlp.tokenizer.mask_token) == 1
                test_sent = masked_context.replace(
                    nlp.tokenizer.mask_token, mask_pred)
                if query != rule: assert test_sent != context
                cand_ginza[(query, mask_pred, rule)].append((context, test_sent))
        query2sents[query] = cand_ginza

    sents = []
    for query, cand_ginza in query2sents.items():
        sents.extend([list(sum(s, ())) for s in cand_ginza.values()])
    sents = set([l for s in sents for l in s])
    sent_s = "\n".join([s for s in sents])

    dir_name = f"chunk_q2r_esupar_{experiment}"
    os.makedirs(dir_name, exist_ok=True)
    sent_fn = f"{dir_name}/q2r_{experiment}_{begin_chunk}_{end_chunk}.txt"
    with open(sent_fn, "w") as f:
        f.write(sent_s)
    with open(f"{dir_name}/q2r_{experiment}_{begin_chunk}_{end_chunk}_query2sents.pkl", "wb") as f:
        pickle.dump(query2sents, f)
else:
    raise ValueError("unknown op")
