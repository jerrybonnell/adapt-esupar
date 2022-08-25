# from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM, TokenClassificationPipeline, pipeline
from collections import defaultdict
# import torch
import numpy as np
import spacy
import pickle
import glob
import pandas as pd
from tqdm import tqdm
import json
import sys
import os
import re
from pprint import pprint

### FOR DIFFICULT BIGRAMS EXPERIMENT (5.3)

models = ["adapt_norules053122",
          "baseline-bccwj-yasuoka-char-cvg-upos_norules053122",
          "baseline-bert-base-japanese-upos_norules053122",
          "baseline-notaiyo-yasuoka-char-cvg-upos_norules053122"]

query2results_dic = {}
for model in models:
    with open(f"query2results_esupar_{model}.pkl", "rb") as f:
        query2results = pickle.load(f)
    query2results_dic[model] = query2results

for model, q2r in query2results_dic.items():
    print((model,len(q2r)))

model2discovered = {}
model2noparsings = defaultdict(list)
for model in query2results_dic:
    discovered = 0
    no_parsings = 0
    for key in query2results_dic[model]:
        discovered += len(query2results_dic[model][key])
        if len(query2results_dic[model][key]) == 0:
            no_parsings += 1
            model2noparsings[model].append(key)
    model2discovered[model] = (discovered, no_parsings)

# adapt has the most rule predictions for un-predicted terms from other models
model_predothers = []
for model_name in models:
    print(model_name)
    for model, noparsings in model2noparsings.items():
        if model == model_name:
            model_predothers.append(
                (model_name, model, 0, 0, 0))
            continue
        num_in_adapt = len([k for k in noparsings if len(
            query2results_dic[model_name][k]) > 0])
        model_predothers.append(
            (model_name, model, num_in_adapt, len(noparsings), num_in_adapt/len(noparsings)))
        print((model, num_in_adapt, len(noparsings), num_in_adapt/len(noparsings)))
    print("---")


all_adapt_keys = query2results_dic['adapt_norules053122'].keys()
num_no_pred = 0
for key in all_adapt_keys:
    no_pred = True
    for model_name in models:
        # if this is empty then there are no predictions
        if len(query2results_dic[model_name][key]) > 0:
            # some model already predicts something; we're done
            # for this one
            no_pred = False
            break
    if no_pred:
        num_no_pred += 1
print(("all no predictions", num_no_pred, len(
    all_adapt_keys), num_no_pred/len(all_adapt_keys)))

df = pd.DataFrame(model_predothers)
df.columns = ['model', 'other_model', 'found', 'no rules', 'percent']
# use for bar plot visualization of results
df.to_csv("model_predictothers.csv", index=False)
