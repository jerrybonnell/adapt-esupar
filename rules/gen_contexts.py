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

print(sys.argv)
assert len(sys.argv) == 2
begin_chunk, end_chunk, op, experiment, m_path = sys.argv[1].split(" ")
begin_chunk = int(begin_chunk)
end_chunk = int(end_chunk)
op = int(op)
print((begin_chunk, end_chunk, op, experiment, m_path))

# NOTE toggle the config here for each system setting
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
    assert 1 == 2
nlp = pipeline("fill-mask", model=model, top_k=15)
diff_dir = f"diff_esupar_{experiment}_tup"

## generate contexts for misclassified bigram terms
## to be used for MLM predictions
bert_norule = []
# these pickled files come from compare_with_pos.py
print(len(glob.glob(f"{diff_dir}/*.pkl")))
for diff in tqdm(glob.glob(f"{diff_dir}/*.pkl")):
    with open(diff, "rb") as f:
        diff_dic = pickle.load(f)
        for sent, diffs in diff_dic:
            bert_norule.append((sent, diffs))
query2context = defaultdict(list)
# for every adapt string, we have a data structure that
# maps the adapt string to a list of tuples where the
# first index is the sentence the adapt string appears
# in and the second index is a list where the one of the
# two characters in that sentence has been masked
print(len(bert_norule))
for sent, diffs in tqdm(bert_norule):
    """
    從軍人夫
    [((('軍人', '夫'), ('NOUN', 'NOUN'), ('compound', 'root')), (('軍人夫',), ('NOUN',), ('root',)))]
    """
    form_part = []
    pos_part = {}
    for diff in diffs:
        assert len(diff) == 2
        form_part.append([list(diff[0][0]), list(diff[1][0])])
        # pos_part[diff[0][0]] = None
        # we're not saving pos from diff[1] b/c we
        # are assuming diff[0] is more accurate
        pos_part[diff[0][0]] = (diff[0][1], diff[0][2])

    concat_diffs = [d for d in form_part if len("".join(d[0])) == 2
                    and len(d[0]) < len(d[1])
                    and "".join(d[0]) == "".join(d[1])
                    and "KANJI" in [get_word_cat(c) for c in d[1]]]

    concat_diffs = list(set(map(tuple, [map(tuple,lis) for lis in concat_diffs])))
    for adapt, ginza in concat_diffs:
        assert len(adapt) == 1 and "".join(ginza) == adapt[0]
        # place the mask token now
        locs = [(m.start(), m.end()) for m in re.finditer(adapt[0], sent)]
        subs = []
        for start, end in locs:
            i = start
            while i < end:
                subs.append((adapt[0].replace(sent[i], "X"),
                    sent[:i] + nlp.tokenizer.mask_token + sent[i + 1:]))
                i += 1
        assert len(subs) % 2 == 0
        # the annotation will always be from adapt for the sentence
        query2context[adapt[0]].append((sent, subs, pos_part[adapt]))

    print(len(query2context))
    x = np.arange(len(dict(list(query2context.items())).keys()))
    # to be used to parallelize
    chunks = [(l[0], l[-1]) for l in np.array_split(x, 200)]

    with open(f"chunks_{experiment}", "w") as f:
        [f.write(f"{l[0]} {l[1]} 0 {experiment} {model}\n") for l in chunks]
    with open(f"query2context_esupar_{experiment}.pkl", "wb") as f:
        pickle.dump(query2context, f)

