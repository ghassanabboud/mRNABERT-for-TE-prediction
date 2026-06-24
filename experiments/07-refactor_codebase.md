# Experiment 07: Refactor entire codebase
 #### **Code version:** huge refactor and pytests(3e771e9fe6e25e2c15481a67afe3606cc5d22d2f)

## Results and Next Steps

The tests all pass and cover watson-crick bias construction, parsing LinearFold output, nan-aware computation of metrics for the RiboNN dataset. They also verify that the SupervisedDataset -> SupervisedDataCollator flow works. Finally, the `_bio_prior_logged` logic in `MaskedRegressionTrainer` verified that the HFTrainer correctly passes the bio_prior matrix to the model.

I reran simple tests on one fold for the watson-crick bias and no bias and the final performance is comparable.

## Objective 

Make the code more modular and easier to maintain in view of someone picking up the project.


## Status
**COMPLETED** 
- **job names**: `refactor_test_biased_no_bias` and `refactor_test_biased_full`

## Expected outcomes
- _Deliverables_: new version of code.
- _output directory_: `outputs/refactor_tests`
- _decisions to take_: N/A


## Resources required

1 GPU.

## Duration
24.06.2026

## Experiment description

### Motivation

The original codebase was a flat collection of scripts with duplicated training infrastructure (dataset loading, collation, trainer loop, metrics) repeated across files. The goal was to extract all shared logic into proper packages so that each root script becomes a thin, readable entry point and future changes only need to be made in one place.

### What was refactored

**Deleted files that are not relevant to the RiboNN task or old versions**: `run_mlm.py`, `regression.py`, `classification.py`, `mlp_regression.py`, `biased_attention_model.py`, `regression_multilabel.py`, `train_biased_head.py`, `preprocess_RiboNN_data.py`, `crossvalidation_preprocessing.py`.

**New package structure**:

```
finetuning/
    arguments.py    — DataArguments, TrainingArguments (shared HF dataclasses)
    datasets.py     — SupervisedDataset (tokenises sequences, always exposes tx_id)
    collators.py    — SupervisedDataCollator (builds bio-prior bias matrices on-the-fly)
    trainers.py     — MaskedRegressionTrainer (NaN-masked loss, optional bio_prior forwarding)
    metrics.py      — calculate_metric_for_regression, safe_save_model_for_hf_trainer

bias/
    model.py        — BioPriorAttention, mRNABERTWithBioPriorHead (pure PyTorch)
    wc.py           — build_wc_lookup (Watson-Crick scoring tensor)
    linearfold.py   — parse_token_ranges, dotbracket_to_token_pairs, run_linearfold, process_one

utils/
    preprocess.py   — sequence extraction helpers + train/dev/test split entry point
    crossvalidation.py — 10-fold CV split entry point
```

**Root entry points** (thin wrappers):
- `train.py` — standard mRNABERT fine-tuning via `AutoModelForSequenceClassification`
- `train_biased.py` — frozen BERT backbone + bio-prior attention head; supports `--bias no_bias|utr_only|full|linearfold`
- `generate_linearfold_bias.py` — CLI wrapper around `bias/linearfold.py`; accepts a file or directory of CSVs and writes a single `.npz`

**Key design decisions**:
- `bias/` depends only on PyTorch; `finetuning/` may import from `bias/` but not the reverse.
- A single `MaskedRegressionTrainer` serves both training scripts: it pops `bio_prior` from the batch only when present.
- `SupervisedDataset` always includes `tx_id`; HF Trainer drops it via `remove_unused_columns=True` in standard mode, while linearfold mode sets `remove_unused_columns=False` so the collator can pop it before building the batch.

### Test suite

Tests live in `tests/` and are written with [pytest](https://docs.pytest.org/en/7.4.x/). They cover:

- **`test_wc_lookup.py`** — `build_wc_lookup`: nuc-nuc, nuc-codon, codon-codon scores, CLS/SEP rows/cols are zero, `utr_only` mode.
- **`test_collator.py`** — `SupervisedDataCollator`: CLS/SEP bias is zero for all sequences in a batch; bio-prior matrix shape is `(L+2, L+2)`; padded positions have zero bias; WC spot-checks for nuc-nuc, nuc-codon, codon-codon token pairs; end-to-end `SupervisedDataset → SupervisedDataCollator → bio_prior` pipeline for both WC and LinearFold modes.
- **`test_linearfold.py`** — `parse_token_ranges` and `dotbracket_to_token_pairs`: single-nuc tokens, codon tokens, mixed tokens, intra-codon pairs discarded, pair counts 1–3. Integration tests (require `--linearfold`) verify that `run_linearfold` and `process_one` produce correctly shaped and typed outputs on real sequences.
- **`test_metrics.py`** — `calculate_metric_for_regression`: perfect predictions, NaN masking per label, `mean_TE` NaN-masking logic, all-NaN sequences excluded from `mean_TE`, 3D logit reshape, `label_names` in output keys, numerical cross-check against scipy/sklearn.

To run the full suite:

```bash
pytest tests/
```

The LinearFold integration tests are skipped by default. To run them, pass the path to the executable:

```bash
pytest tests/ --linearfold /your/path/to/linearfold/executable
```

### Further verification

The pytests verify that atomic components behave as expected. The MaskedRegressionTrainer was set to print on the first batch whether a bio-prior matrix was present and getting passed to the model. This is to verify that the bio-prior is not somehow being dropped by the HF Trainer.

The following script acts as a test run to see if the performance of model fine-tuning is retained:

```bash
#!/bin/bash
#SBATCH --job-name=refactor_test_biased_no_bias
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/refactor_tests/test_biased_no_bias_bigger_run/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
cd /scratch/izar/gabboud/mRNABERT

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9

export JOB_NAME=test_biased_no_bias_bigger_run

mkdir -p outputs/refactor_tests/test_biased_no_bias_bigger_run

python train_biased.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME} \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --output_dir outputs/${JOB_NAME} \
    --num_train_epochs 20 \
    --eval_steps 100 \
    --save_steps 100 \
    --warmup_steps 150 \
    --logging_steps 10 \
    --report_to wandb \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true \
    --num_heads 8 \
    --num_bio_layers 1 \
    --freeze_backbone true \
    --bias no_bias
```


## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

