from collections import defaultdict
import numpy as np
import pickle
import glob
from tqdm import tqdm
import os
import re
from rich.pretty import pprint
import sys

# NOTE toggle system setting here
CONFIG = 4
if CONFIG == 0:
    model = "../../jerrybonnell053122/adapt_yasuoka_char_cvg_50"
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

# toggle option to examine bigram differences that are common
# across all models
COMMON_DIFF_ANALYSIS = True

assert COMMON_DIFF_ANALYSIS or (not COMMON_DIFF_ANALYSIS and CONFIG == 0)

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

# print(s2form['一波は活波を起して大變を來らす時である。'])

# compare everything in terms of ADAPT-ESUPAR labelings
with open(f"query2context_esupar_adapt_norules053122.pkl", "rb") as f:
    query2context = pickle.load(f)

# combine the chunks together into a master dictionary
master_query2sents = {}
print(f"chunk_q2r_esupar_{experiment}/*query2sents.pkl")
print(len(glob.glob(f"chunk_q2r_esupar_{experiment}/*query2sents.pkl")))
for fn in tqdm(glob.glob(f"chunk_q2r_esupar_{experiment}/*query2sents.pkl")):
    with open(fn, "rb") as f:
        master_query2sents.update(pickle.load(f))

### NOTE from query2context_compare.py
with open("common_diffs.pkl", "rb") as f:
    common_keys = pickle.load(f)

query2results = {}
query2results_notbest = {}
query2rules2context = {}
query2counts = {}
skipped = []
for query, cand_ginza in tqdm(master_query2sents.items()):
    # NOTE toggle
    if COMMON_DIFF_ANALYSIS:
        if query not in common_keys:
            continue
    query2counts[query] = len(cand_ginza)
    rules2context = defaultdict(list)
    cand_ginza_score = defaultdict(list)

    if COMMON_DIFF_ANALYSIS:
        adapt_pos = None
        adapt_deprel = None
        pos_set = set()
        deprel_set = set()
        # NOTE
        # we assume ADAPT-ESUPAR parsing is most accurate and base our
        # comparisons of other models with respect to its annotations
        # [('饗庭篁村',
        #  [('X村', '饗庭[MASK]村'), ('篁X', '饗庭篁[MASK]')], (('PROPN',), ('root',)))]
        for q2c_sent, mask, pos in query2context[query]:
            pos_set.add(pos[0][0])
            deprel_set.add(pos[1][0])
        # special cases where there is conflict in POS annotation; these do not
        # comprise majority of cases, so skip these for now
        if len(pos_set) > 1 or len(deprel_set) > 1:
            skipped.append(query)
            continue
        # the assumption by this point is that, if POS annotation is consistent
        # with respect to ADAPT-ESUPAR, then that POS given by ADAPT-ESUPAR can
        # be used carte blanche when checking UD improvement in sentences from
        # baseline models
        adapt_pos = pos_set.pop()
        adapt_deprel = deprel_set.pop()
        assert len(pos_set) == 0 and len(deprel_set) == 0
        assert adapt_pos is not None and adapt_deprel is not None

    # recall cand_ginza from mlm_pred.py
    for (query, mask_pred, rule), sent_tup_lis in cand_ginza.items():
        for org_sent, test_sent in sent_tup_lis:
            #### NOTE toggle this
            if not COMMON_DIFF_ANALYSIS:
                ## use this region of code if not comparing adapt against
                ## other models; really this can only be used under config=0
                # setting
                # get the correct annotation from adapt
                adapt_pos = None
                adapt_deprel = None
                for q2c_sent, mask, pos in query2context[query]:
                    if q2c_sent == org_sent:
                        adapt_pos = pos[0][0]
                        adapt_deprel = pos[1][0]
                        break
            assert adapt_pos is not None and adapt_deprel is not None
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
        # mask_scores_ginza[(query, mask_pred, rule)] = np.mean(scores)
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

