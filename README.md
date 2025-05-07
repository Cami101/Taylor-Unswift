This is the adaption code of Padé Approximation of LLMs base on the paper Taylor Unswift: Secured Weight Access for Large Language Models via Taylor Expansion.


## Dependency

```angular2html
numpy==1.26.4
torch==2.2.0
datasets==2.17.1
accelerate==0.29.2
scikit-learn==1.4.2
peft==0.10.0
transformers==4.38.1
evaluate==0.4.3
ipdb==0.13.13
```


## Test PadeMLP on the wikitext-2, truthul qa, and mmlu datasets:

```bash 
python protection/llama_MN2.py 
python protection/llama_MN4.py
or a simple example run:
python protection/llama_wikitext_test.py
```

Results: 
With Padé approximation of m = n = 4, we achieves 10x+ latency while closely approximates the original Llama's capability.
