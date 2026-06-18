# Experiment 01: reproduce initial finetuning results
 #### **Code version:** remove manual num_labels entry, make mean_TE main regression metric (fb0f5420cc8011a80a86fc9305c4a8c29054cfeb)

## Results and Next Steps

At 1024 max model length, including only the 5'UTR and the CDS yielded better results than the entire sequence at $R^ 2$ of 0.652 compared to 0.644 (marginal improvement). This means that the UTR 3' is not contributing much to the performance of the model. However, doing only the 5' UTR yields 0.476, meaning the cds is important. I did not try CDS only and will try it in the next big scale experiment.

I decided not to test model max_length of 3066 after all because training would be too slow. We can inly accomodate a batch of 2 on the V100 global mem which would lead to training of 1 hour per epoch without evaluation. Instead, max_lenght of 2044 is a good compromise that will demonstrate my point: increased accuracy from more information about the CDS and not about inclusion of the 3'UTR. length of 2044 can accomodate batch sizes of 4 so 30 minuters per epoch without evaluation. That's without mentioning very noisy training that would result from batches of 2. 

For the next step I need 10-fold CV of these results to report more robustly. I will launch CV runs. 

## Objective 

This first experiment will formalize all the experiments I have done on mRNABERT up until now to reproduce the results in a documented way. 

The objective is to fine-tune mRNABERT as before on one specific division of the datasets. I focus on fine-tuning the entire model because initial investigations showed that only tuning the prediction head would lead to low $R^2$ of 0.33. Full-finetuning gives much better results. 

I compare fine-tuning the model on different parts of the sequence to evaluate the contribution of different regions.

First, UTR 5'only versus UTR 5' + CDS versus CDS only at 1024 max model length should show that the interaction between the 5'UTR and the CDS is essential. Then, Comparing UTR 5' + CDS at 3066 max model length versus full transcript at 3066 max model length shows whether the increased performance at 3066 is due to inclusion of the 3'UTR or just the ability to capture longer-range interactions between the 5'UTR and the CDS.

An experiment on fine-tuning the model only on a window of 600 nt around the start codon will determine whether that context is enough.

Then a script can be created for 10-fold CV that will be used consequently in all experiments. 


## Status
**In-Progress** 
- **finetuning runs at 1024 max model length**:
    - full sequence: finetune_HMEC_data (erroneously named), 3051456
    - 5'UTR only: utr5_1024, 3065496
    - 5'UTR + CDS: utr5_cds_1024, 3065497
- **finetuning runs at 3066 max model length**:
    - full sequence: 
    - 5'UTR only: utr5_3066, 3052485
    - 5'UTR + CDS: utr5_cds_3066, 3052584 
- **finetuning runs around the start codon**:
    - start codon window of 600 nt: start_codon_600nt, 3065504
## Expected outcomes
- _Deliverables_: $R^2$ to compare between results.
- _output directory_: `utr5_cds_*`, `utr5_only_*`, `full_*` in `outputs/`
- _decisions to take_:which modes and lengths to use for CV and future experiments.

## Resources required

1 GPU.

## Duration
15.06.2026

## Experiment description

Try to re-run the fine-tuning of mRNABERT on one division of the RiboNN dataset. Focus only on non-freezing. I run the following experiments:
- full sequence at 1024 max model length: 82.65% of sequences cannot fit the entire sequence into the model including 3' UTR
- 5'UTR only at 1024 max model length: 1% of sequences cannot fit their entire 5' UTR into the model
- 5'UTR + CDS at 1024 max model length: 25% of sequences cannot fit their entire 5' UTR and CDS into the model
- start codon window of 600 nt at 400 max model length: I evaluate whether fine-tuning the model only focused on the start codon region can match the results. all of these truncated sequences can fit obviously
- 5'UTR + CDS at 3066 max model length: all sequences can fit into the model
- full sequence at 3066 max model length: 32.66% of sequences still cannot fit into the model because of long 3' UTRs



Once validated, create a script for 10-fold CV.

### script for finetuning full sequence at 1024 max model length

```bash
#!/bin/bash
#SBATCH --job-name=finetune_HMEC_data
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/reproduce_initial_finetuning/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/full_val_fold_8_test_fold_9

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=reproduce_initial_finetuning
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

python regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
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
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true
```

### script for finetuning utr5 only at 1024 max model length

```bash
#!/bin/bash
#SBATCH --job-name=utr5_1024
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/utr5_only_1024/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/utr5_only_val_fold_8_test_fold_9

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=utr5_only_1024
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

python regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
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
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true
```

### script for finetuning utr5 + cds at 1024 max model length

```bash
#!/bin/bash
#SBATCH --job-name=utr5_cds_1024
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/utr5_cds_1024/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=utr5_cds_1024
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

python regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
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
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true
```

### script for finetuning utr5 + cds at 3066 max model length

```bash
#!/bin/bash
#SBATCH --job-name=utr5_cds_3066
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/utr5_cds_3066/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=utr5_cds_3066
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

python regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
    --model_max_length 3066 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --output_dir outputs/${JOB_NAME} \
    --num_train_epochs 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 150 \
    --logging_steps 10 \
    --report_to wandb \
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true

```

### script for finetuning on 600 nt window around start codon at 400 max model length

```bash
#!/bin/bash
#SBATCH --job-name=start_codon_600nt
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=outputs/start_codon_600nt/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/start_codon_window_600nt_val_fold_8_test_fold_9

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=start_codon_600nt
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

python regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
    --model_max_length 400 \
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
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true
```
### script for finetuning full sequence at 3066 max model length

```bash
#!/bin/bash
#SBATCH --job-name=full_3066
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=outputs/full_3066/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/full_val_fold_8_test_fold_9
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=full_3066
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"


torchrun --nproc_per_node=2 regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
    --model_max_length 3066 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --output_dir outputs/${JOB_NAME} \
    --num_train_epochs 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 150 \
    --logging_steps 10 \
    --report_to wandb \
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true
```

### script for finetuning 5' UTR + CDS at 3066 max model length

```bash
#!/bin/bash
#SBATCH --job-name=utr5_cds_3066
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=outputs/utr5_cds_3066/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
#python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
cd /scratch/izar/gabboud/mRNABERT


# fine-tuning data
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export JOB_NAME=utr5_cds_3066
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false


#max_steps is a hard cutoff that overrides the number of epochs, good for testing
#that everything works before

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"


torchrun --nproc_per_node=2 regression_multilabel.py \
    --data_path ${DATA_PATH} \
    --run_name ${JOB_NAME}\
    --model_max_length 3066 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --output_dir outputs/${JOB_NAME} \
    --num_train_epochs 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 150 \
    --logging_steps 10 \
    --report_to wandb \
    --freeze_base false \
    --early_stopping_patience 20 \
    --early_stopping_threshold 0.001 \
    --overwrite_output_dir true

```
## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

