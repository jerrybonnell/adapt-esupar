import mailbox
import os
import sys
import subprocess
import pickle
from rich.progress import track
from rich.pretty import pprint

UD_JAPANESE_MODERN_TEXT = "eval/modern_sents.txt"
UD_JAPANESE_MODERN_URL = "https://github.com/UniversalDependencies/UD_Japanese-Modern/blob/master/ja_modern-ud-test.conllu?raw=true"

def gen_modern_sents():
    import requests
    contents = requests.get(UD_JAPANESE_MODERN_URL).text.split("\n")
    os.makedirs("eval", exist_ok=True)
    with open("eval/ja_modern-ud-test.conllu", "w") as f:
        f.write("\n".join(contents))
    conllu = [l.strip() for l in contents if "# text =" in l]
    sents = [s.split("# text =")[1].strip() for s in conllu]
    sents = ["".join(s.split()) for s in sents]
    sents = [f"{s}\n" for s in sents]
    print(len(sents))
    with open(UD_JAPANESE_MODERN_TEXT, "w") as f:
        f.writelines(sents)

if os.path.exists(UD_JAPANESE_MODERN_TEXT):
    with open(UD_JAPANESE_MODERN_TEXT, "r") as f:
        modern_sents = [l.strip() for l in f.readlines()]
    print(f"loaded {UD_JAPANESE_MODERN_TEXT}")
else:
    print(f"generating {UD_JAPANESE_MODERN_TEXT}..")
    gen_modern_sents()
    with open(UD_JAPANESE_MODERN_TEXT, "r") as f:
        modern_sents = [l.strip() for l in f.readlines()]
print(f"ud japanese modern: {len(modern_sents)} sents")


def evaluate_ginza():
    ginza_out = os.popen(f'ginza -d < {UD_JAPANESE_MODERN_TEXT}')\
        .read().split("\n")
    with open("eval/ginza_modern.conllu", "w") as f:
        f.write("\n".join(ginza_out))

def evaluate_esupar(model):
    print(model)
    import esupar
    esupar_out = []
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    nlp = esupar.load(model)
    for sent in track(modern_sents):
        e = str(nlp(sent))
        e = f"# text = {sent}\n{e}\n"
        esupar_out.append(e)

    if "/" in model:
        model = model.split("/")[-1].replace("-","_")
    pprint(model)
    with open(f"eval/esupar_modern_{model}.conllu", "w") as f:
        f.writelines(esupar_out)
    return f"eval/esupar_modern_{model}.conllu"

def conllu_with_gold(system, gold="eval/ja_modern-ud-test-esupar.conllu"):
    # usage: conll18_ud_eval.py [-h] [--verbose] [--counts] gold_file system_file
    print(system)
    assert os.path.exists(system) and os.path.exists(gold)
    sub2scores = {}
    command = f'python evaluation_script/conll18_ud_eval.py --verbose {gold} {system}'
    print(command)
    p = subprocess.Popen([command],
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    for line in p.stdout.readlines():
        decoded = line.decode()
        if "UPOS" in decoded or "UFeats" in decoded or "UAS" in decoded \
            or "LAS" in decoded or "CLAS" in decoded or "MLAS" in decoded \
            or "BLEX" in decoded:
            parsed = [l.strip() for l in decoded.split("|")]
            sub2scores[parsed[0]] = parsed[1:]
    assert len(sub2scores) != 0
    return {system : sub2scores}


def remove_feats_from_conllu(inp="eval/ja_modern-ud-test.conllu"):
    with open(inp, "r") as f:
        inp_conllu = [l.strip() for l in f.readlines()]

    #inp_conllu = inp_conllu.split("\n")
    for i in range(len(inp_conllu)):
        if "\t" in inp_conllu[i]:
            inp_conllu[i] = inp_conllu[i].split("\t")
            inp_conllu[i][2] = "_"
            inp_conllu[i][4] = "_"
            inp_conllu[i][5] = "_"
            inp_conllu[i][8] = "_"

    for i in range(len(inp_conllu)):
        if type(inp_conllu[i]) == list:
            inp_conllu[i] = "\t".join(inp_conllu[i])
    inp_conllu = "\n".join(inp_conllu)
    base_inp_name = inp.split(".")[0]
    with open(f"{base_inp_name}-esupar.conllu", "w") as f:
        f.writelines(f"{inp_conllu}\n")

def process_esupar_models():
    full_dic = {}
    # NOTE due to copyright of TAIYO and BCCWJ corpora, we are unable to
    # release these model files; results are shown in the paper
    for m in ["jerrybonnell053122/baseline_bccwj_yasuoka_char_cvg_upos",
              "jerrybonnell053122/baseline_notaiyo_yasuoka_char_cvg_upos",
              "jerrybonnell/baseline-bert-base-japanese-upos",
              "jerrybonnell053122/adapt_yasuoka_char_cvg_50_upos"]:
        pprint(m)
        md = conllu_with_gold(evaluate_esupar(m))
        full_dic = full_dic | md

    with open("esupar_modern_results_053122.pkl", "wb") as f:
        pickle.dump(full_dic, f)


def format_results():
    with open("esupar_modern_results_053122.pkl", "rb") as f:
        esupar = pickle.load(f)
    remove_feats_from_conllu("eval/ginza_modern.conllu")
    ginza = conllu_with_gold("eval/ginza_modern-esupar.conllu")
    models = esupar | ginza
    d = {'model': [], 'UPOS': [], 'UAS': [], 'LAS': [], 'CLAS': [],
         'MLAS': [], 'BLEX': []}
    for m in models:
        if "ginza" not in m:
            n = "_".join(m.split(".")[0].split("_")[2:])
            print(n)
        else:
            n = "ginza"
        d['model'].append(n)
        # collect F1 scores
        d['UPOS'].append(models[m]['UPOS'][2])
        d['UAS'].append(models[m]['UAS'][2])
        d['LAS'].append(models[m]['LAS'][2])
        d['CLAS'].append(models[m]['CLAS'][2])
        d['MLAS'].append(models[m]['MLAS'][2])
        d['BLEX'].append(models[m]['BLEX'][2])
    pprint(d)
    with open("meiroku_eval_053122.pkl", "wb") as f:
        pickle.dump(d, f)
    print("meiroku_eval_053122.pkl")

#### example usage for ginza evaluation
remove_feats_from_conllu()
evaluate_ginza()
remove_feats_from_conllu("eval/ginza_modern.conllu")
pprint(conllu_with_gold("eval/ginza_modern.conllu"))

#### we provide the incantation for evaluation of
#### adapt-esupar and baseline methods
# process_esupar_models()
# format_results()
