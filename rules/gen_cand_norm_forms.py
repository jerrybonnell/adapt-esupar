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

# NOTE toggle system setting here
CONFIG = 0
if CONFIG == 0:
    model = "jerrybonnell053122/adapt_yasuoka_char_cvg_50"
    experiment = "adapt_norules053122"
elif CONFIG == 1:
    model = "KoichiYasuoka/bert-base-japanese-char-extended"
    experiment = "esupar_norules053122"
elif CONFIG == 2:
    model = "KoichiYasuoka/bert-base-japanese-char-extended"
    experiment = "baseline-bert-base-japanese-upos_norules053122"
elif CONFIG == 3:
    model = "jerrybonnell053122/baseline_notaiyo_yasuoka_char_cvg"
    experiment = "baseline-notaiyo-yasuoka-char-cvg-upos_norules053122"
elif CONFIG == 4:
    model = "jerrybonnell053122/baseline_bccwj_yasuoka_char_cvg"
    experiment = "baseline-bccwj-yasuoka-char-cvg-upos_norules053122"
else:
    assert 1==2
nlp = pipeline("fill-mask", model=model, top_k=15)
diff_dir = f"diff_esupar_{experiment}_tup"

with open("chunks_q2r_all_esupar/sent_unq.txt", "r") as f:
    sent_unq = [l.strip() for l in tqdm(f.readlines())]

s2form = {}
s2pos = {}
s2deprel = {}
for fn in tqdm(glob.glob("chunk_s2f_esupar/s2f_*.pkl")):
    with open(fn, "rb") as f:
        for sent, (form, pos, deprel) in pickle.load(f).items():
            assert len(form) == len(pos) == len(deprel)
            s2form[sent] = "$$$".join(form)
            s2pos[sent] = "$$$".join(pos)
            s2deprel[sent] = "$$$".join(deprel)

# compare everything in terms of ADAPT-ESUPAR labelings
with open(f"query2context_esupar_adapt_norules053122.pkl", "rb") as f:
    query2context = pickle.load(f)

master_query2sents = {}
print(f"chunk_q2r_esupar_{experiment}/*query2sents.pkl")
print(len(glob.glob(f"chunk_q2r_esupar_{experiment}/*query2sents.pkl")))
for fn in tqdm(glob.glob(f"chunk_q2r_esupar_{experiment}/*query2sents.pkl")):
    with open(fn, "rb") as f:
        master_query2sents.update(pickle.load(f))

### NOTE toggle; from query2context_compare.py
with open("common_diffs.pkl", "rb") as f:
    common_keys = pickle.load(f)

query2results = {}
query2results_notbest = {}
query2rules2context = {}
query2counts = {}
skipped = []
for query, cand_ginza in tqdm(master_query2sents.items()):
    # NOTE toggle
    if query not in common_keys:
        continue
    query2counts[query] = len(cand_ginza)
    rules2context = defaultdict(list)
    cand_ginza_score = defaultdict(list)

    adapt_pos = None
    adapt_deprel = None
    pos_set = set()
    deprel_set = set()
    # NOTE
    # we assume ADAPT-ESUPAR parsing is most accurate and base our
    # comparisons of other models with respect to its annotations
    for q2c_sent, mask, pos in query2context[query]:
        pos_set.add(pos[0][0])
        deprel_set.add(pos[1][0])
    if len(pos_set) > 1 or len(deprel_set) > 1:
        skipped.append(query)
        continue
    adapt_pos = pos_set.pop()
    adapt_deprel = deprel_set.pop()
    assert len(pos_set) == 0 and len(deprel_set) == 0
    assert adapt_pos is not None and adapt_deprel is not None

    for (query, mask_pred, rule), sent_tup_lis in cand_ginza.items():
        for org_sent, test_sent in sent_tup_lis:
            org_form = s2form[org_sent].split("$$$")
            assert ''.join(org_form) == org_sent
            test_form = s2form[test_sent].split("$$$")
            test_pos = s2pos[test_sent].split("$$$")
            test_deprel = s2deprel[test_sent].split("$$$")
            assert ''.join(test_form) == test_sent
            assert adapt_pos is not None and adapt_deprel is not None
            # if the rule 'また' appears in the FORM then it appears as
            # one unit and so, by assumption, transformation was successful
            if rule in test_form \
                    and adapt_pos == test_pos[test_form.index(rule)] \
                    and adapt_deprel == test_deprel[test_form.index(rule)]:
                rules2context[(query, mask_pred, rule)].append(org_sent)
                cand_ginza_score[(query, mask_pred, rule)].append(1)
            else:
                cand_ginza_score[(query, mask_pred, rule)].append(0)

    mask_scores_ginza = {}
    for (query, mask_pred, rule), scores in cand_ginza_score.items():
        mask_scores_ginza[(query, mask_pred, rule)] = np.mean(np.sign(scores))
    mask_scores_ginza_best = {k: v for k,
                              v in mask_scores_ginza.items() if v == 1}
    mask_scores_ginza_notbest = {k: v for k,
                                 v in mask_scores_ginza.items() if 0 <= v < 1}
    query2results[query] = mask_scores_ginza_best
    query2results_notbest[query] = mask_scores_ginza_notbest
    query2rules2context[query] = rules2context

### collect generated candidate normalized forms into a series of pickled data structures
### of most importantance is the query2results dictionary data structure
print(f"query2results_esupar_{experiment}.pkl")
print(len(query2results))
with open(f"query2results_esupar_{experiment}.pkl", "wb") as f:
    pickle.dump(query2results, f)
with open(f"query2results_notbest_esupar_{experiment}.pkl", "wb") as f:
    pickle.dump(query2results_notbest, f)
with open(f"query2results_esupar_rulecontexts_{experiment}.pkl", "wb") as f:
    pickle.dump(query2rules2context, f)
with open(f"skipped_keys_{experiment}.pkl", "wb") as f:
    pickle.dump(skipped, f)

