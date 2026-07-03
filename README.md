# Improving mRNABERT's translation efficiency prediction with structural priors


This repository builds upon the original [mRNABERT codebase](https://github.com/yyly6/mRNABERT). 
It extends its evaluation on predicting mRNA translation efficiency of ultra-long sequences from the [RiboNN dataset](https://github.com/Sanofi-Public/RiboNN). Mainly, it investigates whether incorporating structural priors during finetuning can improve translation efficiency prediction. It also highlights the features learnt by mRNABERT upon finetuning. For information on results, kindly refer to the associated report. Refer to the original [mRNABERT README](https://github.com/yyly6/mRNABERT) for more information about the original model architecture, pre-training and other applications of mRNABERT.

TODO: Add a link to the report once it is available.


## Contents

- [Introduction](#introduction)
- [Create Environment with Conda](#create-environment-with-conda)
- [Testing](#testing)
- [Pre-processing RiboNN dataset](#pre-processing-ribonn-dataset)
- [Fine-tuning on RiboNN dataset](#fine-tuning)
- [Running inference using a fine-tuned model](#running-inference-using-a-fine-tuned-model)
- [References](#references)


## Introduction

`mRNABERTwithBioPriorHead`is this project's main model. It attaches `BioPriorAttention` modules to the original mRNABERT backbone. These modules are standard self-attention modules matching the hidden dimension and multi-head setup of the original mRNABERT model. However, they support adding a pre-computed bias term to steer the model towards attending to certain pairs.

Two types of pre-computed biases are supported:
1) Watson-Crick bias: biases are calculated based on base-pairing rules. A score of 3 is assigned to G-C, 2 to A-T, and 1 to G-T pairs. For pairs involving codons, bias is the sum of all-to-all pairwise scores. For example,  an ATG - TAA pair gets a score of 2 (A-T) + 2 (T-A) + 2 (T-A) + 1 (G-T) = 7. This approach is inspired by [ERNIE-RNA](https://www.nature.com/articles/s41467-025-64972-0).
2) LinearFold bias: secondary structure of mRNA is predicted with [LinearFold](https://github.com/LinearFold/LinearFold) and predicted pairs get a non-zero bias term. Nucleotide-nucleotide pairs and nucleotide-codon pairs get a maximum bias of 1 while codon-codon pairs get a maximum bias of 3. 

## Create Environment with Conda

```bash
# create and activate a virtual python environment
conda create -n mrnabert python=3.8
conda activate mrnabert

# install required packages
pip install -r requirements.txt
pip uninstall triton
```

## Testing

A test suite developed using [pytest](https://docs.pytest.org/en/7.4.x/) verifies the Watson-Crick and LinearFold bias calculations that are the focus of this project. It also verifies the computation of metrics as care needs to be taken to handle `NaN` entries in the RiboNN dataset. To run the tests, run the following from the root of the repository:

```bash
pytest tests/
```

The `generate_linearfold_bias.py` requires that a valid LinearFold binary be installed and its path passed to the script. Four tests in the suite verify that the binary provided yields the expected outputs. While skipped by default, these tests can be run through the following command:

```bash
pytest tests/ --linearfold /your/path/to/linearfold/executable
```


## Pre-processing RiboNN dataset

Download the RiboNN translation efficiency table provided as supplementary data of [Zheng et al.](https://www.nature.com/articles/s41587-025-02712-x#Sec21) through this command:

```bash
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-025-02712-x/MediaObjects/41587_2025_2712_MOESM3_ESM.xlsx human_RiboNN.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-025-02712-x/MediaObjects/41587_2025_2712_MOESM4_ESM.xlsx mouse_RiboNN.xlsx
```

`preprocess_one_split.py` and `preprocess_all_cv_splits.py` scripts generate the RiboNN dataset splits that can be consumed by the model's tokenizer. Splits match those in RiboNN's original work and are indexed from 0 to 9. 


```bash
# one split, keeping the entire sequence
python preprocess_one_split.py --data_path human_RiboNN.xlsx --sequence_mode full --val_fold 8 --test_fold 9 --output_dir processed_data_RiboNN/one_split_full/

# all ten splits, keeping only the 5' UTR region of the mRNA
python preprocess_all_cv_splits.py --data_path human_RiboNN.xlsx --sequence_mode utr5_only
--output_dir processed_data_RiboNN/all_splits_utr5_only/

```

sequence_mode is one of `full`, `cds_only`, `utr5_only`, `utr3_only`, `utr5_cds` to conduct ablation studies on different regions of mRNA. The script will generate three CSV files `train.csv`, `test.csv` and `dev.csv`. If finetuning or running inference on a model using LinearFold bias, Linearfold must be installed and its predictions pre-computed to be passed to the model.

```bash
#only on a single file, using 4 workers for multi-core processing.
python generate_linearfold_bias.py processed_data_RiboNN/one_split_full/train.csv -o processed_data_RiboNN/one_split_full/train_linearfold_bias.npz --num_workers 4 --linearfold /path/to/linearfold/executable

#on all three files in directory
python generate_linearfold_bias.py processed_data_RiboNN/one_split_full/ -o processed_data_RiboNN/one_split_full/all_seq_linearfold_bias.npz --linearfold /path/to/linearfold/executable
```

## Fine-tuning on RiboNN dataset

`train.py` script fine-tunes the base mRNABERT model with no additional `BioPriorAttention` modules. It adds a simple one-layer feedforward head to predict translation efficiency. `train_biased.py` script fine-tunes the `mRNABERTwithBioPriorHead` model with either Watson-Crick or LinearFold bias. In both cases, the base mRNABERT model is pulled from [Huggingface](https://huggingface.co/YYLY66/mRNABERT) as `YYLY66/mRNABERT`. 

```bash
# fine-tuning entire mRNABERT model without BioPriorAttention modules
# truncating sequences to 1024 tokens
# logging results to wandb under run name finetune_entire_model_one_split
python train.py \
    --data_path processed_data_RiboNN/one_split_full \
    --run_name finetune_entire_model \
    --report_to wandb \
    --output_dir outputs/finetune_entire_model \
    --model_max_length 1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --warmup_steps 150 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 10 \


# fine-tuning mRNABERTwithBioPriorHead model with LineaFold bias
# freezing mRNABERT backbone to only train the BioPriorAttention modules and feedforward head
python train_biased.py \
    --data_path processed_data_RiboNN/one_split_full \
    --run_name finetune_lf_biased_model \
    --report_to wandb \
    --output_dir outputs/finetune_lf_biased_model \
    --model_max_length 1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --warmup_steps 150 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 10 \
    --num_heads 8 \
    --num_bio_layers 1 \
    --bias linearfold \
    --linearfold_bias_file processed_data_RiboNN/one_split_full/train_linearfold_bias.npz \
    --freeze_backbone true
```

## Running inference using a fine-tuned model

`predict.py` runs inference using models trained with `train.py` while `predict_biased.py` runs inference using models trained with `train_biased.py`. Example input files are provided in `inference_data/example_inference/`. 

```bash
# example inference using a LineaFold-biased model. example_inference_short.npz was generated
# from example_inference_short.csv using generate_linearfold_bias.py
  python predict_biased.py \
      --checkpoint_path outputs/finetune_lf_biased_model \
      --input_csv inference_data/example_inference/example_inference_short.csv \
      --linearfold_bias_file inference_data/example_inference/example_inference_short.npz \
      --output_dir predictions/example_inference_lf_bias
```

## Analysis and Plotting

This project also investigates whether finetuning mRNABERT on the RiboNN dataset makes the model learn translation-relevant features.

- `study_AUG_insertion.py`: investigates whether the model learns the negative effect of upstream AUGs on translation initiation.
- `study_codon_optimality.py`: Organisms have optimal codons that are translated more efficiently than their synonymous counterparts due to higher tRNA abundance. This script investigates whether introducing synonymous mutations that use optimal codons increases the predicted translation efficiency.
- `study_attention_ss_correlation.py`: investigates whether the attention scores of the mRNABERT backbone correlate with LinearFold-predicted secondary structure of the mRNA. This would make introduction of LinearFold bias redundant. 

All plotting scripts can also be found in `figure_scripts`.


## References

The code in this repository builds on that of the original [mRNABERT model](https://github.com/yyly6/mRNABERT) and of the [RiboNN model](https://github.com/Sanofi-Public/RiboNN). It also heavily relies on the [transformers library from Huggingface](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling).

