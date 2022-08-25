
## Prerequisites

We provide a `requirements.txt` file for the conda environment used in our experiments. You can replicate the environment on your machine using the incantation:

```
conda create --name adaptesupar --file requirements.txt
```

## Copyright

There are copyright issues with the corpora used in our research (TAIYO and BCCWJ) that prevent us from possibly revealing any training data used. For this reason, we are unable to directly release any model files or training data.

## Flagging differences in UD output

We use the script `compare_with_pos.py` to generate discrepancies in UD output between ESUPAR and each of the four models tested.

## Generating masked contexts for misclassified bigrams

The script `gen_contexts.py` generates masked contexts for sentences that contain misclassified bigram terms, which are used for the MLM objective. This is done for each of the four systems tested. These are serialized to disk as a dictionary mapping bigram terms to a three-element tuple: the sentence that contains the bigram, a list of masked contexts for the sentence, and the UPOS and DEPREL annotation for the bigram term according to ADAPT-ESUPAR.

## MLM predictions

The (parallelized) script `mlm_pred.py` generates MLM predictions from the masked contexts and substitutes the mask token (`[MASK]`) with each of the masked predictions to form a series of original/test sentence pairs that are to be submitted to a pretrained LM for annotation to see if UD annotation improvement is brought by the test sentence.

The script `prep_annotate.py` is an intermediary step that reorganizes the sentences to be annotated over all systems into a single file so that the following annotation step can be trivially parallelized using GNU parallel. The script `run_sent_chunk.py` (together with `run_sents.sh`) implements that annotation work in parallel and produces UD annotations for each original/test sentence pair using a pretraiend LM (i.e., ESUPAR). The annotations are serialized to disk as a dictionary.

The script `query2context_compare.py` implements the experimental results shown in Section 5.2 using the masked context dictionary.

## Candidate normalized form discovery

The script `gen_cand_norm_forms.py` scores potential rules using the original/test sentence pairs found in `mlm_pred.py`. We use an indicator variable to score any pair: `1` if the UD annotations for the test sentence correspond to the annotaton given by ADAPT-ESUPAR (in terms of FORM, UPOS, and DEPREL), and `0` otherwise. A substitution is designated a candidate normalized form if and only if the test sentences corresponding to it consistently bring improvement in UD, i.e., all instances are scored `1`. The results are serialized to disk as a dictionary mapping the candidate normalized form to its score.

The script `query2results_compare.py` implements the experimental results shown in Section 5.3 using the candidate normalized form dictionary.

