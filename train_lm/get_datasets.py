import pickle
from tqdm import tqdm
import random

mlm_cvg_hack = 10

def datasets_from_pkl():
    t_examples = []
    b_examples = []
    g_examples = []
    bs_examples = []
    # NOTE due to copyright of the the target corpora, we are unable to provide
    # access to these data structures
    with open("../jp/processed/taiyo_train.pkl", "rb") as f:
        t_sents = ["".join(t[0]) for t in pickle.load(f)]
    with open("../jp/processed/bccwj_train.pkl", "rb") as f:
        b_sents = ["".join(t[0]) for t in pickle.load(f)]
    with open("../jp/processed/gsd_train.pkl", "rb") as f:
        g_sents = ["".join(t[0]) for t in pickle.load(f)]
    with open(
            "../../../../bccwj/bccwj_scripts/bccwj_noncore_sampled.pkl", "rb") as f:
        b_samp_sents = pickle.load(f)
    print((len(t_sents), len(b_sents), len(g_sents), len(b_samp_sents)))
    assert len(b_samp_sents) == len(t_sents)
    # Han and Eistenstein perscribe an equal amount of text from the source
    # and target domains for domain tuning; fill in the hole for the source
    # corpus with a remainder of non-core text from BCCWJ
    random.seed(2022)
    bccwj_filler_sents = random.sample(b_samp_sents,
                                       len(t_sents) - len(b_sents) - len(g_sents))
    print(len(bccwj_filler_sents))
    assert len(t_sents) == len(b_sents) + \
        len(g_sents) + len(bccwj_filler_sents)

    for t in tqdm(t_sents):
        for j in range(mlm_cvg_hack):
            t_examples.append(t)
    for b in tqdm(b_sents):
        for j in range(mlm_cvg_hack):
            b_examples.append(b)
    for g in tqdm(g_sents):
        for j in range(mlm_cvg_hack):
            g_examples.append(g)
    for bs in bccwj_filler_sents:
        for j in range(mlm_cvg_hack):
            bs_examples.append(bs)
    print((len(t_examples), len(b_examples), len(g_examples), len(bs_examples)))

    with open("adapt_yasuoka_char_cvg_train.txt", "w") as f:
        f.write("\n".join(t_examples + b_examples + g_examples))

    print(len(b_examples + g_examples))
    with open("baseline_notaiyo_yasuoka_char_cvg_train.txt", "w") as f:
        i = 0
        for ex in b_examples + g_examples:
            i += 1
            f.write(f"{ex}\n")
    assert i == len(b_examples + g_examples)
    print(len(b_examples + g_examples + bs_examples))
    with open("baseline_bccwj_yasuoka_char_cvg_train.txt", "w") as f:
        i = 0
        for ex in b_examples + g_examples + bs_examples:
            i += 1
            cleaned = ex.replace('\n','').strip()
            f.write(f"{cleaned}\n")
    print(i)
    assert i == len(b_examples + g_examples + bs_examples)

datasets_from_pkl()
