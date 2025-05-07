import torch
from torch import nn

from datasets import load_dataset
import ipdb, copy, sys, os
import numpy as np
import time
import json
import math

sys.path.append("../")
sys.path.append("./")

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pade_expansion.llama import model_pade_expansion
import metrics
# from util import model_expand, print_model_parameters, freeze_model_parameters, zero_model_parameters, ViTBase2Large

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
dtype = "torch.bfloat16"

# ———— Prepare tokenizer & dataset ————
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", cache_dir="/scratch"
)
# — Load & expand approximated model —
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    cache_dir="/scratch",
    torch_dtype=eval(dtype),
    attn_implementation="eager"
).to(device)
model_pade_expansion(
    model,
    "./output/llama-3-8b-hf-pileval-hidden-states/hidden-states-"
    + dtype
    + "-n-20000-len-4096.pth.tar",
    select_dim=2048,
    grad_order=4, # m = n = 2
    grad_order_min=2,
    delta_hidden_state_thd=2.5,
    expand_layer=None,
)
print(metrics.compute_ppl(model, tokenizer))
print(metrics.compute_QA_score(model, tokenizer))
print(metrics.compute_MMLU_score(model, tokenizer))