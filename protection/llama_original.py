import torch
from torch import nn

from datasets import load_dataset
import ipdb, copy, sys, os
import numpy as np
import time
import json
import math
from tqdm import tqdm

sys.path.append("../")
sys.path.append("./")

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import evaluate
import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
dtype = "torch.bfloat16"

# ———— Prepare tokenizer & dataset ————
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", cache_dir="/scratch"
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                             cache_dir="/scratch",
                                             torch_dtype=eval(dtype),
                                             # use_flash_attention_2=True
                                             attn_implementation = "eager"
                                             ).to(device)
print(metrics.compute_ppl(model, tokenizer))
print(metrics.compute_QA_score(model, tokenizer))
print(metrics.compute_MMLU_score(model, tokenizer))