import os
import sys
import glob
import pickle
from rich.console import Console
from rich.pretty import pprint
from tqdm import tqdm

console = Console()

rules_conllu = glob.glob("../../Rules2UD/annotation_041222_esupar/*.conllu")

CONFIG = 4
if CONFIG == 0:
    adapt_conllu = glob.glob(
        "../../Rules2UD/annotation_053122_adapt_yasuoka_char_cvg_50_upos/*.conllu")
    model = "jerrybonnell053122/adapt_yasuoka_char_cvg_50"
    experiment = "adapt_norules053122"
elif CONFIG == 1:
    adapt_conllu = glob.glob(
        "../Rules2UD/annotation_041222_esupar/*.conllu")
    model = "KoichiYasuoka/bert-base-japanese-char-extended"
    experiment = "esupar_norules053122"
elif CONFIG == 2:
    adapt_conllu = glob.glob(
        "../../Rules2UD/annotation_053122_baseline-bert-base-japanese-upos/*.conllu")
    model = "KoichiYasuoka/bert-base-japanese-char-extended"
    experiment = "baseline-bert-base-japanese-upos_norules053122"
elif CONFIG == 3:
    adapt_conllu = glob.glob(
        "../../Rules2UD/annotation_053122_baseline_notaiyo_yasuoka_char_cvg_upos/*.conllu")
    model = "jerrybonnell053122/baseline_notaiyo_yasuoka_char_cvg"
    experiment = "baseline-notaiyo-yasuoka-char-cvg-upos_norules053122"
elif CONFIG == 4:
    adapt_conllu = glob.glob(
        "../../Rules2UD/annotation_053122_baseline_bccwj_yasuoka_char_cvg_upos/*.conllu")
    model = "jerrybonnell053122/baseline_bccwj_yasuoka_char_cvg"
    experiment = "baseline-bccwj-yasuoka-char-cvg-upos_norules053122"
else:
    assert 1 == 2

print((model, experiment, adapt_conllu[0]))

test_set = glob.glob("../../jp/resources/test/*")
test_set = [os.path.split(fn)[-1].split(".")[0] for fn in test_set]
print(len(test_set))

def form_helper(adapt_conllu, rule_conllu,
    adapt_pos, rule_pos, adapt_deprel, rule_deprel):
    # form: [adaptabert, rules] <- TODO make this a dictionary so easier to understand
    #console.log(f"rule : {rule_conllu}")
    #console.log(f"adapt: {adapt_conllu}")
    # bool list to check if we have added that index to the list yet
    # check if this index has been checked already
    rule_bool_list = [False] * len(rule_conllu)
    adapt_bool_list = [False] * len(adapt_conllu)

    #console.log(f"rule : {rule_pos}")
    #console.log(f"adapt: {adapt_pos}")

    big_diff_list = [] # final output
    diff_list_adapt = []
    diff_list_rules = []
    pos_list_adapt = []
    pos_list_rules = []
    deprel_list_adapt = []
    deprel_list_rules = []
    i, j = 0, 0
    # will stop when one of them has reached the entire length
    while i < len(rule_conllu) or j < len(adapt_conllu):
        #console.log(f"rule: {rule_conllu[i]}")
        #console.log(f"adapt: {adapt_conllu[j]}")
        #console.log(f"i: {i} j: {j}")

        # most cases go here; everything is equal
        # because the diff lists for each are empty, everything is in a
        # clean state; so just increment for the next iteration
        # in other words, there is nothing to do here
        if rule_conllu[i] == adapt_conllu[j] and\
                len(diff_list_adapt) == 0 and\
                len(diff_list_rules) == 0:
            #console.log("if")
            i += 1
            j += 1
        # rule word is split across multiple blocks
        else:
            #console.log("else")
            # if the boolean is false, then this index has not been
            # inserted into the diff_list_rules yet
            if not rule_bool_list[i]:
                diff_list_rules.append(rule_conllu[i])
                pos_list_rules.append(rule_pos[i])
                deprel_list_rules.append(rule_deprel[i])
                rule_bool_list[i] = True
            if not adapt_bool_list[j]:
                diff_list_adapt.append(adapt_conllu[j])
                pos_list_adapt.append(adapt_pos[j])
                deprel_list_adapt.append(adapt_deprel[j])
                adapt_bool_list[j] = True
            rule_str = "".join(diff_list_rules)
            adapt_str = "".join(diff_list_adapt)
            #console.log(f"rule_str: {rule_str}")
            #console.log(f"adapt_str: {adapt_str}")
            if len(adapt_str) > len(rule_str):
                #console.log("increment i")
                i += 1
            elif len(adapt_str) < len(rule_str):
                #console.log("increment j")
                j += 1
            else:
                #console.log("else else")
                i += 1
                j += 1
                big_diff_list.append(
                    ((tuple(diff_list_adapt),
                      tuple(pos_list_adapt),
                      tuple(deprel_list_adapt)),
                     (tuple(diff_list_rules),
                      tuple(pos_list_rules),
                      tuple(deprel_list_rules)
                     )))
                diff_list_adapt = []
                diff_list_rules = []
                pos_list_adapt = []
                pos_list_rules = []
                deprel_list_adapt = []
                deprel_list_rules = []
    """
    [(('軍人', '夫'), ('軍人夫'))]
    [(['軍人', '夫'],['NOUN','NOUN']), (['軍人夫'],['NOUN'])]
    """
    return big_diff_list


def compare_form(adapt_conllu, rule_conllu):
    res_dic = []
    for adapt_sent, rule_sent in zip(adapt_conllu, rule_conllu):
        adapt_form = [l[1] for l in adapt_sent[1:]]
        rule_form = [l[1] for l in rule_sent[1:]]
        adapt_pos = [l[3] for l in adapt_sent[1:]]
        rule_pos = [l[3] for l in rule_sent[1:]]
        adapt_deprel = [l[7] for l in adapt_sent[1:]]
        rule_deprel = [l[7] for l in rule_sent[1:]]
        try:
            res = form_helper(adapt_form, rule_form,
                adapt_pos, rule_pos, adapt_deprel, rule_deprel)
            if len(res) > 0:
                res_dic.append((adapt_sent[0][9:],res))
        except:
            raise ValueError(f"adapt:{adapt_form} rule:{rule_form}")
    return res_dic


def load_conllu(conllu_fname):
    with open(conllu_fname, "r") as f:
        conllu = [l.split("\n") for l in f.read().split("\n\n")]
    conllu = conllu[:-1]
    for i in range(len(conllu)):
        conllu[i] = [
            l.split("\t") if "# text" not in l else l for l in conllu[i]]
    return conllu




rules_conllu = [c for c in rules_conllu if os.path.split(
    c)[-1].split(".")[0] in test_set]
adapt_conllu = [c for c in adapt_conllu if os.path.split(
    c)[-1].split(".")[0] in test_set]

assert len(rules_conllu) == len(adapt_conllu)
# dummy file in test set (named "empty") because of git stuff
assert len(rules_conllu) == len(test_set) - 1

print(len(rules_conllu))
for rule_conllu_fname, adapt_conllu_fname in\
    tqdm(zip(sorted(rules_conllu), sorted(adapt_conllu)),
         total=len(rules_conllu)):
    if adapt_conllu_fname in []:
        continue
    assert os.path.split(
        rule_conllu_fname)[-1] == os.path.split(adapt_conllu_fname)[-1]
    rule_conllu = load_conllu(rule_conllu_fname)
    adapt_conllu = load_conllu(adapt_conllu_fname)
    res = compare_form(adapt_conllu, rule_conllu)

    fname = os.path.split(rule_conllu_fname)[-1].split(".")[0]
    fname = f"{experiment}_{fname}.pkl"

    dir_name = f"diff_esupar_{experiment}_tup"
    os.makedirs(dir_name, exist_ok=True)
    with open(f"{dir_name}/{fname}", "wb") as f:
        pickle.dump(res, f)
