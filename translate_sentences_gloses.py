"""
Fine-tuning mT5 or SinBERT (BERT2BERT) for Sinhala to Sinhala Sign Language (SSL) Gloss Translation.

Requirements:
    pip install torch transformers datasets evaluate sentencepiece accelerate rouge_score sacrebleu pandas openpyxl

Usage:
    python translate_sentences_gloses.py --data_path "Sinhala Sentences.xlsx" --model_name NLPC-UOM/SinBERT-large
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import evaluate

# Setup logging/metrics
def compute_metrics(eval_preds, tokenizer, metric_bleu, metric_rouge):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # Calculate BLEU
    bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Calculate ROUGE
    # ROUGE expect single string references per prediction
    rouge_labels = [label[0] for label in decoded_labels]
    rouge_result = metric_rouge.compute(predictions=decoded_preds, references=rouge_labels)

    # Exact Match Accuracy (Gloss level)
    exact_matches = sum([1 for p, l in zip(decoded_preds, rouge_labels) if p.upper() == l.upper()])
    exact_match_acc = exact_matches / len(decoded_preds)

    return {
        "bleu": bleu_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rougeL": rouge_result["rougeL"],
        "exact_match_accuracy": exact_match_acc,
    }

def preprocess_function(examples, tokenizer, max_input_length, max_target_length, prefix):
    # Prepare inputs with prefix
    inputs = [prefix.format(sentence=str(s)) for s in examples["sinhala_sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Prepare labels
    labels = tokenizer([str(s) for s in examples["ssl_gloss"]], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(args):
    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
    
    if args.data_path.endswith(('.xlsx', '.xls')):
        print(f"Loading Excel dataset: {args.data_path}")
        df = pd.read_excel(args.data_path)
    else:
        print(f"Loading CSV dataset: {args.data_path}")
        df = pd.read_csv(args.data_path)

    # Ensure column names match if they are different in CSV
    # Standardizing columns
    df = df.rename(columns={args.src_col: "sinhala_sentence", args.tgt_col: "ssl_gloss"})
    
    # Simple split
    dataset = Dataset.from_pandas(df)
    
    # Split: 80% train, 20% (val + test)
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    # Split the 20% into 10% validation and 10% test
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        "train": train_testvalid["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"]
    })
    
    print(f"Dataset split: {len(dataset_dict['train'])} train, {len(dataset_dict['validation'])} validation, {len(dataset_dict['test'])} test")

    # 2. Initialize Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if "SinBERT" in args.model_name or "bert" in args.model_name.lower():
        print(f"Detected encoder-only model style. Initializing EncoderDecoderModel using {args.model_name} as both encoder and decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name, args.model_name)
        
        # Configure the model for the specific types of sequences we are generating
        # BERT uses cls_token for BOS and sep_token for EOS usually
        model.config.decoder_start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        
        # Ensure vocab sizes match for BERT2BERT setup
        model.config.vocab_size = model.config.encoder.vocab_size
        
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 3. Preprocess Dataset
    prefix = (
        "Translate the following Sinhala sentence into Sinhala Sign Language (SSL) gloss sequence. "
        "SSL gloss uses uppercase English-like words that represent individual signs. "
        "Follow SSL grammar rules (often topic-comment or SOV structure).\n\n"
        "Sinhala Sentence: {sentence}\n\n"
        "SSL Gloss:"
    )

    tokenized_datasets = dataset_dict.map(
        lambda x: preprocess_function(x, tokenizer, args.max_input_length, args.max_target_length, prefix),
        batched=True,
        remove_columns=dataset_dict["train"].column_names
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none", # Disable wandb/others for simplicity, unless env is set
        logging_steps=10,
    )

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Metrics
    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, metric_bleu, metric_rouge),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if args.early_stopping else None
    )

    # Train
    print("Starting training...")
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    elif resume and resume.lower() == "none":
        resume = None
    
    trainer.train(resume_from_checkpoint=resume)

    # Final Evaluation on Test set
    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Results: {results}")

    # Save best model
    trainer.save_model(args.output_dir)
    print(f"Best model saved to {args.output_dir}")

    return model, tokenizer

def run_inference(model, tokenizer, sentence, max_input_length=128):
    prefix = (
        "Translate the following Sinhala sentence into Sinhala Sign Language (SSL) gloss sequence. "
        "SSL gloss uses uppercase English-like words that represent individual signs. "
        "Follow SSL grammar rules (often topic-comment or SOV structure).\n\n"
        "Sinhala Sentence: {sentence}\n\n"
        "SSL Gloss:"
    )
    
    input_text = prefix.format(sentence=sentence)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=False)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune mT5 for Sinhala to SSL Gloss Translation")
    
    # Hyperparameters & Paths
    parser.add_argument("--data_path", type=str, default="Sinhala Sentences.xlsx", help="Path to the dataset")
    parser.add_argument("--src_col", type=str, default="Sentence", help="Name of source (Sinhala) column")
    parser.add_argument("--tgt_col", type=str, default="Sign Language", help="Name of target (SSL Gloss) column")
    parser.add_argument("--model_name", type=str, default="NLPC-UOM/SinBERT-large", help="Hugging Face model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results_sinbert_ssl", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_input_length", type=int, default=128, help="Max input sequence length")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max target sequence length")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'True' to resume from latest")
    
    args = parser.parse_args()

    # Run training
    model, tokenizer = train_model(args)

    # Example Inference
    test_sentence = "\u0db8\u0d9c\u0dda \u0db1\u0db8 \u0d9a\u0dca\u0dbd\u0dd2\u0db1\u0dca\u0d9c\u0db1\u0dca" # Example: My name is Clingan
    gloss = run_inference(model, tokenizer, test_sentence)
    print(f"\nExample Inference:")
    print(f"Sentence: {test_sentence}")
    print(f"Generated Gloss: {gloss}")
