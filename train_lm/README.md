
## Prerequisites

We provide a `requirements.txt` file for the conda environment used in our experiments. You can replicate the environment on your machine using the incantation:

```
conda create --name adaptesupar --file requirements.txt
```

For interaction with GINZA, we used as a separate environment as per the [official documentation](https://github.com/megagonlabs/ginza#runtime-environment). We provide another requirements file if you wish to use our configuration (in `requirements_ginza.txt`).

## Copyright

There are copyright issues with the corpora used in our research (TAIYO and BCCWJ) that prevent us from possibly revealing any training data used. For this reason, we are unable to directory release any model files or training data. However, assuming the datasets are available, we provide the scripts used for LM training and evaluation.

## Dataset preparation

We use the script `process.py` to partition the corpora (TAIYO and BCCWJ) into training, dev, and testing sets. We then use the script `get_datasets.py` to prepare the training text files that are input to each of the four systems tested.

## LM training

Please see the shell script `run.sh`.

## Evaluation scripts

Please see the script `evaluate_modern.py`. An example usage of this script is provided for interfacing with GINZA, as shown in the paper.
