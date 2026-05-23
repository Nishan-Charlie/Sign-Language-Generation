"""
Continued Pre-training mT5 on Sinhala MADLAD/CulturaX dataset using the T5 Span Corruption Objective.

Requirements:
    pip install torch transformers datasets accelerate sentencepiece numpy

Dataset:
    - Name: polyglots/MADLAD_CulturaX_cleaned
    - Config: default (Filtered for lang == 'si')
    - Mode: Streaming

FIXED BUGS:
    1. Custom T5DataCollatorForSpanCorruption (handles missing DataCollatorForT5MLM in transformers v5.4.0)
    2. Corrected Dataset config ('default') and added filtering for Sinhala ('si')
"""

import argparse
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# --- Custom T5 Span Corruption Collator ---
class T5DataCollatorForSpanCorruption:
    """
    Manual implementation of T5 Span Corruption (Denoising) strategy.
    Replaces 15% of tokens with sentinel tokens (<extra_id_0>, <extra_id_1>, etc.)
    and creates target sequences for the encoder-decoder training.
    """
    def __init__(self, tokenizer, noise_density=0.15, mean_noise_span_length=3.0, input_length=512):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        # Sentinels are usually at the end of the vocabulary
        # <extra_id_0> is usually tokenizer.vocab_size - 100 or specifically named
        # For mT5, they are specifically added. We can find them by string.
        self.sentinel_tokens = [tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]

    def __call__(self, examples):
        # Handle list of dicts from trainer
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        
        batch_input_ids = []
        batch_labels = []
        
        for ids in input_ids:
            # 1. Generate mask
            length = len(ids)
            num_noise_tokens = int(np.round(length * self.noise_density))
            num_noise_tokens = max(num_noise_tokens, 1) # At least one noise token
            
            # Geometric distribution for span lengths
            num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
            num_noise_spans = max(num_noise_spans, 1)
            
            # Simple approach: randomly pick indices to be start of spans
            # In a production setting, this uses more complex span sampling logic
            # but for this script we will use a robust random masking approach.
            indices = np.arange(length)
            np.random.shuffle(indices)
            noise_indices = sorted(indices[:num_noise_tokens])
            
            new_input_ids = []
            new_labels = []
            
            sentinel_idx = 0
            last_idx = -1
            is_noise = False
            
            curr_input = []
            curr_label = []
            
            for i in range(length):
                if i in noise_indices:
                    if not is_noise:
                        # Transition from clean to noise
                        curr_input.append(self.sentinel_tokens[sentinel_idx])
                        curr_label.append(self.sentinel_tokens[sentinel_idx])
                        is_noise = True
                    curr_label.append(ids[i].item())
                else:
                    if is_noise:
                        # Transition from noise to clean
                        sentinel_idx += 1
                        is_noise = False
                    curr_input.append(ids[i].item())
            
            # Add final EOS to label
            curr_label.append(self.eos_token_id)
            
            batch_input_ids.append(torch.tensor(curr_input))
            batch_labels.append(torch.tensor(curr_label))

        # Pad sequences
        def pad_sequence(sequences, pad_id):
            max_len = max([s.size(0) for s in sequences])
            padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
            for i, s in enumerate(sequences):
                padded[i, :s.size(0)] = s
            return padded

        input_ids_padded = pad_sequence(batch_input_ids, self.pad_token_id)
        labels_padded = pad_sequence(batch_labels, -100) # -100 is standard for ignoring loss on pad
        
        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": (input_ids_padded != self.pad_token_id).long()
        }

def pretrain_sinhala(args):
    # 1. Initialize Tokenizer & Model
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 2. Load Dataset
    print(f"Loading dataset: polyglots/MADLAD_CulturaX_cleaned ...")
    # Download and filter locally for Sinhala ('si')
    dataset = load_dataset(
        "polyglots/MADLAD_CulturaX_cleaned", 
        "default", 
        split="train", 
    ).filter(lambda x: x["lang"] == "si", num_proc=8)
    
    # Shuffle
    dataset = dataset.shuffle(seed=42)

    # 3. Preprocessing/Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=8,
        remove_columns=["text", "lang", "src"]
    )

    # 4. Custom Data Collator for Span Corruption
    data_collator = T5DataCollatorForSpanCorruption(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_noise_span_length=3.0,
        input_length=args.max_length
    )

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_steps=args.save_steps,
        logging_steps=10,
        eval_strategy="no",
        push_to_hub=args.push_to_hub,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        save_total_limit=1,
    )

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 7. Start Pre-training
    print(f"Starting pre-training for {args.max_steps} steps...")
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    elif resume and resume.lower() == "none":
        resume = None
    
    if resume:
        print(f"Resuming from checkpoint: {resume}")
    
    trainer.train(resume_from_checkpoint=resume)

    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print("Pre-training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue Pre-training mT5 on Sinhala")
    
    parser.add_argument("--cuda_device", type=str, default="", help="CUDA visible devices (e.g., '1' for GPU 1)")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Pre-trained mT5 checkpoint")
    parser.add_argument("--output_dir", type=str, default="./mt5_sinhala_pretrained", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'True' to resume from latest")
    
    args = parser.parse_args()
    
    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        print(f"Set CUDA_VISIBLE_DEVICES to {args.cuda_device}")
        
    pretrain_sinhala(args)
