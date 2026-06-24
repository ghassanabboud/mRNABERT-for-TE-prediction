from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    num_train_epochs: int = field(default=20, metadata={"help": "Total number of training epochs to perform."})
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=64)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=250)
    eval_steps: int = field(default=250)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=300)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=0.0001)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="r2_mean_TE")
    greater_is_better: bool = field(default=True)
    output_dir: str = field(default="output_gena")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: Optional[str] = field(default="wandb")
    overwrite_output_dir: bool = field(default=True)
    log_level: str = field(default="info")
    eval_and_save_results: bool = field(default=True)
    early_stopping_patience: int = field(default=5, metadata={"help": "Stop after this many evals with no improvement."})
    early_stopping_threshold: float = field(default=0.0, metadata={"help": "Minimum improvement to count as an improvement for early stopping."})
