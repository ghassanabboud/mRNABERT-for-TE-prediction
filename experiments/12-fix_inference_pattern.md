# Experiment 12: Fix pattern for label-less inference
 #### **Code version:** save config.json for mRNABERTWithBioPriorHead and reload from checkpoint. Use id2label for labeled inference (3289ca0f8778e19e7395c4cb46dcd378d3b7d641)

## Results and Next Steps

Smoke tests run withut issues and output csv files with correct column names. For the big equivalence test, I get a value of R2 for mean TE on the test set of 0.637. Before refactoring I got one of 0.635. So we can assume that the refactoring did not break anything and that the difference is stochastic.

NOTE: this means that the current `predict.py` and `predict_biased.py` scripts are now no longer compatible with checkpoints saved before this refactoring, especially all checkpoints in `cv_*` folders from which the main results are derived.

## Objective 

I created `predict.py` and `predict_biased.py` to run inference on sequences that do not have labels. I then noticed 2 issues with the current setup:
- at no point does `mRNABERTWithBioPriorHead` saved information about its configuration and architecture and there was no easy way to create a model from checkpoint. This led to the used needing to re-enter into the inference script the arch parameteres that were used to train the model. Error-prone and annoying.
- there was no info about the name of the cell types in the checkpoint, so label-less inference created columns with generic names like `predicted_0`.

This experiment fixes these issues then runs tests to ensure equivalence.



## Status
**COMPLETED** 
- **job names**: train_label_config_handling, train_biased_label_config_handling, predict, whole_training_run_lf_bias_1_layer
## Expected outcomes
- _Deliverables_: A check that the new code functions properly and outputs CSVs with the correct column names.
- _output directory_: all smoke tests under `outputs/label_config_handling/`: `test_train` `test_train_biased`, `predict_finetuned` and `predict_lf_bias`. For the big equivelance run, `outputs/whole_training_run_lf_bias_1_layer`
- _decisions to take_: N/A


## Resources required

1 GPU.

## Duration
03.07.2026

## Experiment description


Main Changes:
- `mRNABERTWithBioPriorHead` now can be initiated from a checkpoint saved by `train_biased.py`. On top of saving the model weights, `train_biased.py` also saves a `bio_prior_config.json` file that contains num_heads, num_bio_layers, bias mode, base model, and cell-type names. Hence, `predict_biased.py` no longer needs to pass these parameters to the model, it can just read them from the checkpoint. if `predict_biased.py` detects that the model was trained with `--bias linearfold`, it also requires a `--linearfold_bias_file` to be passed, which must be generated for the sequences in `--input_csv` (via `generate_linearfold_bias.py`).
- `train.py` reads the cell-type names from `SupervisedDataset` then adds them to the `id2label` attribute of the config object, saved as `config.json`. `predict.py` can use them to name the columns of the output CSV.

For testing, I do two smoke-tests, one using `train.py` and `predict.py` and one using `train_biased.py` and `predict_biased.py`. I set max_steps=20 and only verify that the prediction scripts instantiate the model without errors and save CSVs with correct labels


### Smoke tests slurm script example

```bash
#!/bin/bash
#SBATCH --job-name=train_biased_label_config_handling
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=outputs/label_config_handling/test_train_biased/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
cd /scratch/izar/gabboud/mRNABERT

export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export WANDB_PROJECT=mRNABERT-finetuning
export WANDB_LOG_MODEL=true
export WANDB_WATCH=false
export DATA_PATH=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/utr5_cds_val_fold_8_test_fold_9
export LF_BIAS=/scratch/izar/gabboud/mRNABERT/processed_data_RiboNN/all_lf_bias.npz
export HF_HOME=/scratch/izar/gabboud/.cache/huggingface

mkdir -p outputs/label_config_handling/test_train_biased

python train_biased.py \
    --data_path ${DATA_PATH} \
    --run_name train_biased_label_config_handling \
    --model_max_length 1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --output_dir outputs/label_config_handling/test_train_biased \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 20 \
    --logging_steps 5 \
    --report_to wandb \
    --overwrite_output_dir true \
    --num_heads 8 \
    --num_bio_layers 1 \
    --freeze_backbone true \
    --bias linearfold \
    --linearfold_bias_file ${LF_BIAS}
```

```bash
#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/label_config_handling/predict_lf_bias/job_%j.out

eval "$(mamba shell hook --shell bash)"
mamba activate mrnabert
cd /scratch/izar/gabboud/mRNABERT

python predict_biased.py \
    --checkpoint_path outputs/label_config_handling/test_train_biased \
    --input_csv /scratch/izar/gabboud/mRNABERT/processed_data/example_inference/example_inference_short.csv \
    --output_dir outputs/label_config_handling/predict_lf_bias \
    --linearfold_bias_file /scratch/izar/gabboud/mRNABERT/processed_data/example_inference/example_inference_short.npz
```

Then, to verify mathematical equivalence of results, I fully train an instance of the LF biased modelon one fold, test fold 3 val fold 4. 


## Links and references
TO-DO: list here publications, web pages, etc. that contain information relevant to the experiment. 

