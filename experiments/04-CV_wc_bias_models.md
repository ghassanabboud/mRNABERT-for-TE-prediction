# Experiment 04: 10-fold CV on Watson-Crick bias model
 #### **Code version:** initial experiments on bias model variations barplot(fa09a55f5aa1a5d89f1a0ac21cfd2f0eb0be292f)

## Results and Next Steps

WC bias performs worse than no bias in the 10-fold CV.


NOTE FROM 01.07: THIS EXPERIMENT CONTAINS A BUG IN THE WATSON-CRICK BIAS CONSTRUCTION. REFER TO EXPERIMENT 08 FOR THE FIXED VERSION. 

## Objective 

To more rigorously evaluate the effect of Watson-Crick bias as described in [experiment 03](03-wc_bias_model.md), I run 10-fold CV on the most important configurations. This will yield results to report in a boxplot, similar to the ablation study in [experiment 02](02-CV_ablation_studies.md).

## Status
**RUNNING** 
- **job names**:
    - cv_biased_full_1024_frozen_1_layer_wc_bias
    - cv_biased_full_1024_frozen_2_layer_wc_bias
    - cv_biased_full_1024_frozen_3_layer_wc_bias
    - cv_biased_full_1024_frozen_1_layer_no_bias
    - cv_biased_full_1024_frozen_2_layer_no_bias
    - cv_biased_full_1024_frozen_3_layer_no_bias

## Expected outcomes
- _Deliverables_: 
- _output directory_: `cv_biased_full_*` in `outputs/`
- _decisions to take_: boxplot to supplement with LinearFold results later to put in report.


## Resources required

1 GPU.

## Duration
21.06.2026

## Experiment description

All experiments use a 1024 max model length and full sequence mode. The reason I switched to full compared to utr_cds used in [experiment 03](03-wc_bias_model.md) is that I reasoned that maybe some utr3 interaction might show up from introducing bias. Probably not but why would I remove the utr3 anyway. It still uses the same memory because max model length is 1024 and including it doesn't worsen performance as seen in [experiment 02](02-CV_ablation_studies.md), it even betters it in a non-significant way. I freeze all backbones for faster training even though it was shown in [experiment 03](03-wc_bias_model.md) that unfreezing the backbone gives a good performance boost from 0.62 to 0.64.

I compare the full watson-crick bias model to the no bias model, both with 1,2 and 3 Bio-Prior layers. 


### example scripts

all present in `jobs/cv_biased_model/`.


```bash
#!/bin/bash
#SBATCH --job-name=cv_biased_full_1024_frozen_1_layer_no_bias
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=outputs/cv_biased_full_1024_frozen_1_layer_no_bias/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
cd /scratch/izar/gabboud/mRNABERT

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface

BASE_DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/cv_full
OUTPUT_BASE=outputs/cv_biased_full_1024_frozen_1_layer_no_bias

# Map array index to fold directory (sorted order)
FOLD_DIRS=($(ls -d ${BASE_DATA_PATH}/val_fold_* | sort))
FOLD_DIR=${FOLD_DIRS[$SLURM_ARRAY_TASK_ID]}
FOLD_NAME=$(basename ${FOLD_DIR})

VAL_FOLD=$(echo ${FOLD_NAME} | sed 's/val_fold_\([0-9]*\)_test_fold_\([0-9]*\)/\1/')
TEST_FOLD=$(echo ${FOLD_NAME} | sed 's/val_fold_\([0-9]*\)_test_fold_\([0-9]*\)/\2/')

RUN_NAME=cv_biased_full_1024_frozen_1_layer_no_bias_${FOLD_NAME}

mkdir -p ${OUTPUT_BASE}/${FOLD_NAME}


python train_biased_head.py \
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
    --bias no_bias


```

## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

