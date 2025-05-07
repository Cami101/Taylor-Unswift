import math
import torch
from datasets import load_dataset
from tqdm import tqdm

device = "cuda"

# ——— Perplexity helper ———
def compute_ppl(model, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = ds["text"]
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for line in texts:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer(line, return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            # loss is mean NLL per token
            nll = outputs.loss.item() * input_ids.numel()
            total_nll += nll
            total_tokens += input_ids.numel()
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)

# ——— QA dataset helper ———
def compute_QA_score(model, tokenizer):
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    correct = 0
        
    for example in tqdm(ds):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        label = example["mc1_targets"]["labels"].index(1)
        
        scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            
            scores.append(-loss)
        
        prediction = scores.index(max(scores))
        if prediction == label:
            correct += 1
    
    accuracy = correct / len(ds)
    return accuracy

# ——— MMLU dataset helper ———
def compute_MMLU_score(model, tokenizer, subjects=None):
    subjects = ["high_school_mathematics", "high_school_us_history", "computer_security"]
    
    results = {}
    all_correct = 0
    all_total = 0
    
    for subject in subjects:
        print(f"\nEvaluating on MMLU subject: {subject}")
        
        ds = load_dataset("cais/mmlu", subject)
        split = "test" if "test" in ds else "validation"
        if split not in ds:
            print(f"No test or validation split available for {subject}, skipping")
            continue
            
        # Limit to first 100 examples for efficiency if needed
        examples = ds[split]
        if len(examples) > 100:
            examples = examples.select(range(100))
        
        correct = 0
        total = 0
        
        for example in tqdm(examples):
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]  # Integer index of correct answer
            
            # Create prompts for each choice
            choice_letters = ["A", "B", "C", "D"]
            formatted_question = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{choice_letters[i]}. {choice}\n"
            
            scores = []
            
            # Evaluate model on each possible answer
            for i in range(len(choices)):
                choice_letter = choice_letters[i]
                prompt = formatted_question + f"Answer: {choice_letter}"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                
                scores.append(-loss)  # Negative loss as score (higher is better)
            
            # Get model's prediction
            prediction = scores.index(max(scores))
            
            # Check if prediction matches ground truth
            if prediction == answer_idx:
                correct += 1
            total += 1
        
        # Calculate accuracy for this subject
        accuracy = correct / total if total > 0 else 0
        results[subject] = accuracy
        
        all_correct += correct
        all_total += total
        
        print(f"Accuracy on {subject}: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate average accuracy across all subjects
    if all_total > 0:
        results["average"] = all_correct / all_total
        print(f"\nAverage accuracy across all subjects: {results['average']:.4f}")
    
    return results