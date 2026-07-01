import argparse

import pandas as pd
import torch

from utils.analysis import (
    find_utr5_cds_boundaries,
    get_cai,
    get_max_usage_sequence,
    get_min_usage_sequence,
    load_model,
)

TEST_CSV_PATH = "processed_data_RiboNN/cv_full/val_fold_4_test_fold_3/test.csv"

df = pd.read_csv(TEST_CSV_PATH)
#df = df.iloc[:100]
df["truncated_sequence"] = df["sequence"].apply(lambda seq: " ".join(seq.split(" ")))
df["cai"] = df["truncated_sequence"].apply(get_cai)

df.to_csv("./data_with_cai.csv", index=False)