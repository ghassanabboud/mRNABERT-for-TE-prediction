# Experiment 05: LinearFold bias model
 #### **Code version:** integrate linearfold bias results(76aa91da7140141b3e322527227ebac90f691e04)

## Results and Next Steps

Performance is worse than the other variants, from 0.62 to 0.59. One thing previous experiments have shown is that doing this 1-fold test does not always reflect the CV results. Hence I'll still run CV for the tthis variant with different number of layers (1, 2, 3). I need these results for my final boxplot comparing the biased models.


## Objective 

We will now implement a version of the biased model that uses LinearFold predictions to compute the bias matrix instead of a simple Watson-Crick pairing. As usual, an initial investigation on one fold will provide insights before CV.


## Status
**COMPLETED** 
- **job names**: `biased_head_wc_full_1024_frozen_1_layer_lf_bias`, `biased_head_wc_full_1024_frozen_2_layer_lf_bias`, `biased_head_wc_full_1024_frozen_3_layer_lf_bias`

## Expected outcomes
- _Deliverables_: supplement the barplot with the LinearFold bias model results.
- _output directory_: `./outputs/biased_head_wc_full_1024_frozen_1_layer_lf_bias` and `2_layer` and `3_layer` variants.
- _decisions to take_: whether to run CV for that variant.


## Resources required

1 GPU.

## Duration
23.06.2026

## Experiment description

The structure of the biased model does not have to change as it supports any bias matrix given by the batch collator. Here are the implementation choices:
- LinearFold predictions should be pre-computed or else they will dominate training time as they're not GPU-accelerated. The script `generate_linearfold_bias.py` is used to read a CSV file and run LinearFold on all sequences. The dot-bracket notation is transformed into a contact map and saved as a (K,3) array of (pos_i, pos_j, bias_ij) for K non-zero pairs. The contact matrix is very sparse so this is a memory-efficient representation. A contact corresponds to a score of 1. Hence, base-base pairs and base-codon pairs are at most 1 while codon-codon pairs are at most 3. The output is saved in a npz file where contact maps are indexed by transcript ID.
- a special class `TxIdSupervisedDataset` builds a dataset like `SupervisedDataset` but also returns the transcript ID for each sequence. CAREFUL: trainer.args.remove_unused_columns must be set to False so that the HFTrainer does not strip the tx_id from the batch before it reaches the collator. 
- the collator, `LinearFoldDataCollator`, queries, for each batch the transcript IDs and reads the corresponding contact maps from the npz file. It then builds the bias matrix for each sequence in the batch. The bias matrix is (L+2. L+2) where L is the length of the largest sequence in the batch because we have CLS and SEP tokens.
- The rest of the model is uncahnged as it can integrate any bias matrix. 


As usual, I test the model on one fold first before running CV.

### example scripts

all present in `jobs/lf_tests/`.


```bash
#!/bin/bash
#SBATCH --job-name=biased_head_wc_full_1024_frozen_1_layer_lf_bias
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=outputs/biased_head_wc_full_1024_frozen_1_layer_lf_bias/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
cd /scratch/izar/gabboud/mRNABERT

export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/full_val_fold_8_test_fold_9
export BIAS_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/all_lf_bias.npz

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface

export JOB_NAME=biased_head_wc_full_1024_frozen_1_layer_lf_bias

mkdir -p outputs/${JOB_NAME}

# Use --max_steps for a quick sanity check before full training
# --max_steps 50

python train_biased_head.py \
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
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 150 \
    --logging_steps 10 \
    --report_to wandb \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true \
    --num_heads 8 \
    --num_bio_layers 1 \
    --freeze_backbone true \
    --bias linearfold \
    --linearfold_bias_file ${BIAS_PATH}



```

## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

