import re
import os
import sys
import json
import glob
import numpy as np
import pickle
import bs4 as bs
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sys import argv
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint

dataset = sys.argv[1]

def gather_sents_by_article(out_dir="../../../out/*.xml", pkl_name='doc_to_sents_full.pkl'):
    """
    prepares doc_to_sents.pkl pickle by gathering all article sentences
    into a dictionary from document to list of sents
    """
    doc_list = glob.glob(out_dir)

    print(len(doc_list))
    doc_to_sents = {}
    for doc in tqdm(doc_list):
        with open(doc, 'rb') as f:
            soup = bs.BeautifulSoup(f, 'xml')
        tag = soup.find('記事')
        body_lis = []
        for link in soup.find_all('s'):
            body_lis.append(link.get_text().strip())
        # some preprocessing
        body_lis = [''.join(b.split()) for b in body_lis]
        doc_name = doc[doc.rfind("/")+1:doc.rfind(".xml")]
        doc_to_sents[doc_name] = body_lis

    with open(f"resources/{pkl_name}", "wb") as f:
        pickle.dump(OrderedDict(doc_to_sents), f)
    print(f"wrote {pkl_name}")

def write_train_test_chunks(doc2sents_pkl='resources/doc_to_sents_full.pkl'):
    with open(doc2sents_pkl, "rb") as f:
        doc_to_sents = pickle.load(f)
    # kf = KFold(n_splits=5, shuffle=True)
    # 75% train 25% test
    train_indices, test_indices = train_test_split(
        list(doc_to_sents.keys()),
        test_size=0.25, shuffle=True)
    with open("resources/train_indices.txt", "w") as f:
        [f.write(f"{t}\n") for t in train_indices]
    with open("resources/test_indices.txt", "w") as f:
        [f.write(f"{t}\n") for t in test_indices]

def prepare_train_test_set():
    with open("resources/doc_to_sents_full.pkl", "rb") as f:
        doc_to_sents = pickle.load(f)
    with open("resources/train_indices.txt", "r") as f:
        train_indices = [l.strip() for l in f.readlines()]
    with open("resources/test_indices.txt", "r") as f:
        test_indices = [l.strip() for l in f.readlines()]

    print(len(train_indices))
    print(len(test_indices))
    assert len(doc_to_sents) == len(train_indices) + len(test_indices)

    for train_doc_name in train_indices:
        with open(f"resources/train/{train_doc_name}.txt", "w") as f:
            [f.write(f"{l}\n") for l in doc_to_sents[train_doc_name]]
    for test_doc_name in test_indices:
        with open(f"resources/test/{test_doc_name}.txt", "w") as f:
            [f.write(f"{l}\n") for l in doc_to_sents[test_doc_name]]

if dataset == "prepare_taiyo_train_test":
    prepare_train_test_set()

if dataset == "taiyo_train":
    base = "resources/train/"  # "../resources/conair_PPCEME_pos/train/"
    newbase = "processed/"
    data_list = []

    taiyo_files = [
        base+fname for fname in os.listdir(base) if fname.endswith('.txt')]
    for fname in tqdm(taiyo_files):
        # with open(fname, 'rb') as f:
        #     soup = bs.BeautifulSoup(f, 'xml')
        with open(fname, "r") as f:
            lines = f.readlines()
        body_lis = [l.strip() for l in lines]
        # for l in lines:
        #     body_lis.append(l.strip())
        body_lis = [''.join(b.split()) for b in body_lis]
        for sent in body_lis:
            char_tokenized = list(sent)
            # no labels available for taiyo
            pos_labels = ['UNK'] * len(char_tokenized)
            data_list.append((char_tokenized, pos_labels))

    pickle.dump(data_list, open("processed/taiyo_train.pkl", 'wb'))

if dataset == "bccwj_train":
    base = "../../../../bccwj/bccwj/UD_Japanese-BCCWJ/"
    newbase = "processed/"

    data_list = []

    with open(f"{base}ja_bccwj-ud-train.conllu.word", "r") as f:
        bccwj_conllu = f.read().split("\n")
    bccwj = [u.split("\t") for u in bccwj_conllu]

    #sents = []
    #pos_labels = []
    sent = ""
    pos = []
    for line in tqdm(bccwj):
        if len(line) == 10:
            sent += line[1]
            p = [line[3]] if len(line[1])==1 \
                else ["B-"+line[3]]+["I-"+line[3]]*(len(line[1])-1)
            pos.extend(p)
        elif len(line) == 1 and len(sent) > 0:
            assert len(list(sent)) == len(pos)
            data_list.append((list(sent), pos))
            #sents.append(sent)
            #pos_labels.append(pos)
            sent = ""
            pos = []
        elif len(line) == 1 and len(sent) == 0:
            continue
        else:
            raise ValueError("weird line length")


    assert len(data_list) == len([s for s in data_list if len(s[1]) > 0])
    pickle.dump(data_list, open("processed/bccwj_train.pkl", 'wb'))

if dataset == "gsd_train":
    base = "../../../../bccwj/bccwj/UD_Japanese-GSD/"
    newbase = "processed/"

    data_list = []

    with open(f"{base}ja_gsd-ud-train.conllu", "r") as f:
        bccwj_conllu = f.read().split("\n")
    bccwj = [u.split("\t") for u in bccwj_conllu]

    #sents = []
    #pos_labels = []
    sent = ""
    pos = []
    for line in tqdm(bccwj):
        if len(line) == 10:
            sent += line[1]
            p = [line[3]] if len(line[1]) == 1 \
                else ["B-"+line[3]]+["I-"+line[3]]*(len(line[1])-1)
            pos.extend(p)
        elif len(line) == 1 and len(sent) > 0:
            assert len(list(sent)) == len(pos)
            data_list.append((list(sent), pos))
            #sents.append(sent)
            #pos_labels.append(pos)
            sent = ""
            pos = []
        elif len(line) == 1 and len(sent) == 0:
            continue
        else:
            raise ValueError("weird line length")

    assert len(data_list) == len([s for s in data_list if len(s[1]) > 0])
    pickle.dump(data_list, open("processed/gsd_train.pkl", 'wb'))

if dataset == "bccwj_dev":
    base = "../../../../bccwj/bccwj/UD_Japanese-BCCWJ/"
    newbase = "processed/"
    data_list = []

    with open(f"{base}ja_bccwj-ud-dev.conllu.word", "r") as f:
        bccwj_conllu = f.read().split("\n")
    bccwj = [u.split("\t") for u in bccwj_conllu]

    #sents = []
    #pos_labels = []
    sent = ""
    pos = []
    for line in tqdm(bccwj):
        if len(line) == 10:
            sent += line[1]
            p = [line[3]] if len(line[1])==1 \
                else ["B-"+line[3]]+["I-"+line[3]]*(len(line[1])-1)
            pos.extend(p)
        elif len(line) == 1 and len(sent) > 0:
            assert len(list(sent)) == len(pos)
            data_list.append((list(sent), pos))
            #sents.append(sent)
            #pos_labels.append(pos)
            sent = ""
            pos = []
        elif len(line) == 1 and len(sent) == 0:
            continue
        else:
            raise ValueError("weird line length")

    assert len(data_list) == len([s for s in data_list if len(s[1]) > 0])

    pickle.dump(data_list, open("processed/bccwj_dev.pkl", 'wb'))

if dataset == "taiyo_test":
    base = "resources/test/"
    newbase = "processed/"
    data_list = []

    taiyo_files = [
        base+fname for fname in os.listdir(base) if fname.endswith('.txt')]
    for fname in tqdm(taiyo_files):
        with open(fname, "r") as f:
            lines = f.readlines()
        body_lis = [l.strip() for l in lines]
        body_lis = [''.join(b.split()) for b in body_lis]
        for sent in body_lis:
            char_tokenized = list(sent)
            # no labels available for taiyo
            pos_labels = ['UNK'] * len(char_tokenized)
            data_list.append((char_tokenized, pos_labels))

    pickle.dump(data_list, open("processed/taiyo_test.pkl", 'wb'))



