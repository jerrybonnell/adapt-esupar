from collections import defaultdict
import torch
import numpy as np
import spacy
import pickle
import glob
from tqdm import tqdm
import json
import sys
import os
import re
from rich.pretty import pprint

cat_info_map = {}

def load_dictionary():
    with open(f'char.def', 'r') as f:
        cat_info_list = f.read().split('\n')
    cat_info_list = [l.split(' ') for l in cat_info_list][:-1]
    # convert the unicode representation to just an integer
    for line in cat_info_list:
        line[0] = int(line[0], 16)
        if len(line) > 2:
            line[1] = int(line[1], 16)
        else:
            # things strictly one value are added to the map
            cat_info_map[line[0]] = line[1]
    # delete all sublists that are strictly one value
    cat_info_list = [l for l in cat_info_list if len(l) == 3]
    # sort the list
    cat_info_list.sort(key=lambda x: x[0])
    return cat_info_list, cat_info_map


cat_info_list, cat_info_map = load_dictionary()


def get_word_cat(char):
    assert len(char) == 1
    char_value = ord(char)
    # now we do search in the char.def file to determine which
    # category of unknown this char is from (which corresponds
    # to the cat_info_list and cat_info_map ds's)
    if char_value in cat_info_map:
        return cat_info_map[char_value]
    for entry in cat_info_list:
        if char_value >= entry[0] and char_value <= entry[1]:
            return entry[2]
    return 'DEFAULT'  # default is a mandatory category

models = ["adapt_norules053122",
          "baseline-bccwj-yasuoka-char-cvg-upos_norules053122",
          "baseline-bert-base-japanese-upos_norules053122",
          "baseline-notaiyo-yasuoka-char-cvg-upos_norules053122"]

query2context_dic = {}
for model in models:
    fn = f"query2context_esupar_{model}.pkl"
    with open(fn, "rb") as f:
        query2context_dic[model] = pickle.load(f)

for model, q2c in query2context_dic.items():
    print(model, len(q2c))

adapt_keys = list(query2context_dic['adapt_norules053122'].keys())
print(len(adapt_keys))

for model, q2c in query2context_dic.items():
    if model == "adapt_norules053122":
        continue
    for key in q2c:
        if key in adapt_keys:
            adapt_keys.remove(key)

print(adapt_keys[:20])
print(len(adapt_keys))

model2unq = {}
for name in tqdm(models):
    unq_keys = list(query2context_dic[name].keys())
    for model, q2c in query2context_dic.items():
        if model == name:
            continue
        for key in q2c:
            if key in unq_keys:
                unq_keys.remove(key)
    model2unq[name] = unq_keys


unq_set_lis = [set(list(query2context_dic[n].keys())) for n in models]
common_diffs = set.intersection(*unq_set_lis)
print([(len(model2unq[k]),k) for k in model2unq])

with open("common_diffs.pkl", "wb") as f:
    pickle.dump(common_diffs, f)

#### FOR DETERMINING OOV TERMS (5.2)
with open("../../symbolism/baseline_bccwj_yasuoka_char_cvg_train.txt", "r") as f:
    baseline_bccwj = list(set([l.strip() for l in f.readlines() if len(l.strip()) > 0]))

for model_name in models:
    in_v_count = 0
    for key in tqdm(model2unq[model_name]):
        for sent in baseline_bccwj:
            if key in sent:
                in_v_count += 1
                break
    print((model_name, len(model2unq[model_name]), in_v_count, (len(
        model2unq[model_name]) - in_v_count)/len(model2unq[model_name])))

in_v_count = 0
for key in tqdm(common_diffs):
    for sent in baseline_bccwj:
        if key in sent:
            in_v_count += 1
            break
print(("common", len(common_diffs), in_v_count,
       (len(common_diffs) - in_v_count)/len(common_diffs)))

with open("adapt_keys_unique_esupar.pkl", "wb") as f:
    pickle.dump(adapt_keys, f)

### FOR VENN DIAGRAM VISUALIZATION (5.2)
model2diffs = {}
for model in models:
    diff_dir = f"diff_esupar_{model}_tup"
    bert_norule = []

    for diff_fn in tqdm(glob.glob(f"{diff_dir}/*.pkl")):
        with open(diff_fn, "rb") as f:
            diff_dic = pickle.load(f)
            for sent, diffs in diff_dic:
                bert_norule.append((sent, diffs))
    tup_diffs = []
    print(len(bert_norule))
    for sent, diffs in bert_norule:
        for diff in diffs:
            # NOTE for no_pos usage, need to drop the last [0] in below
            # list comprehension when filtering down to just the
            # two-length concat forms
            no_pos = tuple([d[0] for d in diff])
            tup_diffs.append(no_pos)

    tup_diffs = [tuple(list(t)) for t in set(tup_diffs)]
    # NOTE toggle
    tup_diffs = [t for t in tup_diffs if len("".join(t[0])) == 2
                 and len(t[0]) < len(t[1])
                 and "".join(t[0]) == "".join(t[1])
                 and "KANJI" in [get_word_cat(c) for c in "".join(t[0])]]
    model2diffs[model] = tup_diffs
    print(model, len(tup_diffs))

diff2id = {}
intid = 0
for model, tups in model2diffs.items():
    print(model)
    print(tups[:10])
    for tup in tqdm(tups):
        if tup not in diff2id:
            diff2id[tup] = intid
            intid += 1
print(len(diff2id))

model2id = {}
for model, tups in model2diffs.items():
    id_list = []
    for tup in tqdm(tups):
        id_list.append(diff2id[tup])
    print(id_list[:10])
    model2id[model] = id_list

with open("model2id-2char.pkl", "wb") as f:
    pickle.dump(model2id, f)
