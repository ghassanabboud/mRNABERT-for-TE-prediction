# Experiment 08: Rerun Watson-Crick bias without bug
 #### **Code version:** fixed biased models boxplot with new codebase(3ad7c1071980838f53bf88f066fcff06f801a8b2)

## Results and Next Steps

The conclusions do not change, the Watson-Crick bias model still performs the same as No Bias with a non-signficant trend to be worse. The bug was a low bias omission (wobble pairs have bias of 1) and was probably compensated by learning. Anyway it was good to rerun with the correct version.

## Objective 

I noticed that the Watson-Crick bias construction in [experiment 04](04-CV_wc_bias_models.md) was incorrect. the G-T wobble pair was only assigned bias in one direction (G->T) but not the other (T->G). The refactor fixed the bugs and pytests were added to verify. The LinearFold bias and No Bias models are not affected by this bug. I hence rerun the Watson-Crick bias model but I do not expect the results to change because of such a low bias omission (wobble pairs have bias of 1).


## Status
**COMPLETED** 
- **job names**: `cv_FIXED_biased_full_1024_frozen_1_layer_wc_bias` and 2_layer, 3_layer variants

## Expected outcomes
- _Deliverables_: updated outputs and a new version of the bias boxplot
- _output directory_: `outputs/cv_FIXED_biased_full_1024_frozen_2_layer_wc_bias`, `outputs/cv_FIXED_biased_full_1024_frozen_1_layer_wc_bias`, `outputs/cv_FIXED_biased_full_1024_frozen_3_layer_wc_bias`, new boxplot at `figures/r2_scores_cv_bias_FIXED.png`
- _decisions to take_: N/A


## Resources required

1 GPU.

## Duration
01.07.2026

## Experiment description

Same setup as in [experiment 04](04-CV_wc_bias_models.md) but with new codebase. 1024 model max length, 10-fold cv, varying the number of Bio-Prior layers (1,2,3) and freezing the backbone. 


```bash
#!/bin/bash
#SBATCH --job-name=cv_FIXED_biased_full_1024_frozen_2_layer_wc_bias
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=outputs/cv_FIXED_biased_full_1024_frozen_2_layer_wc_bias/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
cd /scratch/izar/gabboud/mRNABERT

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface

BASE_DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/cv_full
OUTPUT_BASE=outputs/cv_FIXED_biased_full_1024_frozen_2_layer_wc_bias

# Map array index to fold directory (sorted order)
FOLD_DIRS=($(ls -d ${BASE_DATA_PATH}/val_fold_* | sort))
FOLD_DIR=${FOLD_DIRS[$SLURM_ARRAY_TASK_ID]}
FOLD_NAME=$(basename ${FOLD_DIR})

VAL_FOLD=$(echo ${FOLD_NAME} | sed 's/val_fold_\([0-9]*\)_test_fold_\([0-9]*\)/\1/')
TEST_FOLD=$(echo ${FOLD_NAME} | sed 's/val_fold_\([0-9]*\)_test_fold_\([0-9]*\)/\2/')

RUN_NAME=cv_FIXED_biased_full_1024_frozen_2_layer_wc_bias_${FOLD_NAME}

mkdir -p ${OUTPUT_BASE}/${FOLD_NAME}

python train_biased.py \
    --data_path ${FOLD_DIR} \
    --run_name ${RUN_NAME} \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --output_dir ${OUTPUT_BASE}/${FOLD_NAME} \
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
    --num_bio_layers 2 \
    --freeze_backbone true \
    --bias full

```


## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

