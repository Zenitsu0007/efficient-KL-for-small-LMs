# Install necessary libraries on Colab
# !pip install pytorch_lightning transformers datasets sentencepiece peft rouge_score nltk

import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(24)

# Fine-tuning with LoRA
class T5FineTunerWithLoRA(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTunerWithLoRA, self).__init__()
        self.save_hyperparameters(hparams)
        
        # Load T5 model and tokenizer (no explicit cache path needed in Colab)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        
        # Set up LoRA
        lora_config = LoraConfig(
            r=4,  # Rank of the low-rank decomposition
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"],  # Apply LoRA to the query and value projection matrices
        )
        
        # Apply LoRA to the T5 model
        self.model = get_peft_model(self.model, lora_config)
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask, 
            labels=labels
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        t_total = (
            (len(self.train_dataloader().dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams, subset_size=100)
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=2)

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams, subset_size=50)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=2)

# Dataset class definition
class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, subset_size=None):
        self.path = os.path.join(data_dir, type_path + '.tsv')
        self.source_column = "sentence1"
        self.target_column = "sentence2"
        self.data = pd.read_csv(self.path, sep="\t").astype(str)
        if subset_size:
            self.data = self.data.sample(n=subset_size, random_state=24).reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, self.target_column]
            input_ = "paraphrase: " + input_ + ' </s>'
            target = target + " </s>"
            tokenized_inputs = self.tokenizer.batch_encode_plus([input_], max_length=self.max_len, padding="max_length", return_tensors="pt", truncation=True)
            tokenized_targets = self.tokenizer.batch_encode_plus([target], max_length=self.max_len, padding="max_length", return_tensors="pt", truncation=True)
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

def get_dataset(tokenizer, type_path, args, subset_size=None):
    return ParaphraseDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length, subset_size=subset_size)

if __name__ == '__main__':
    # Define parameters and cache directory
    args_dict = {
        "data_dir": "data",
        "output_dir": "t5_lora_paraphrase",
        "model_name_or_path": 't5-large',
        "tokenizer_name_or_path": 't5-large',
        "max_seq_length": 256,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "warmup_steps": 0,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_train_epochs": 2,
        "gradient_accumulation_steps": 16,
        "n_gpu": 1,
        "seed": 24,
        "fp_16": False,
    }
    args = argparse.Namespace(**args_dict)

    train_params = {
        "accumulate_grad_batches": args.gradient_accumulation_steps,
        "max_epochs": args.num_train_epochs,
        "precision": 16 if args.fp_16 else 32,
    }

    # Initialize and train the LoRA model
    print("Initialize model with LoRA")
    model = T5FineTunerWithLoRA(args)
    trainer = pl.Trainer(**train_params)
    print("Training model with LoRA")
    trainer.fit(model)
    print("Training finished")

    # -----------------Test-------------------- 
    print("Begin Test")
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    dataset = load_dataset("glue", "mrpc")

    # Set model to evaluation mode
    model.eval()

    smoothie = SmoothingFunction().method1
    alpha = 0.9
    bleu_scores = []
    i_bleu_scores = []
    rouge_l_scores = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def prepare_input(sentence1, sentence2):
        input_text = f"paraphrase: {sentence1} </s>"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
        return input_ids

    count = 0
    for row in dataset['test']:
        count += 1
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        label = row['label']

        input_ids = prepare_input(sentence1, sentence2)
        outputs = model.model.generate(input_ids=input_ids, max_length=256)
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate BLEU score
        reference = [sentence2.split()]
        hypothesis = predicted_text.split()
        bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)
        
        # Calculate iBLEU
        bleu_on_input = sentence_bleu([sentence1.split()], hypothesis, smoothing_function=smoothie)
        i_bleu = alpha * bleu_score - (1 - alpha) * bleu_on_input
        i_bleu_scores.append(i_bleu)

        # Calculate ROUGE-L score
        rouge_l_result = scorer.score(sentence2, predicted_text)
        rouge_l_score = rouge_l_result['rougeL'].fmeasure
        rouge_l_scores.append(rouge_l_score)

        if count >= 100:
            break

    # Print Average BLEU, iBLEU, and ROUGE-L scores
    average_bleu = sum(bleu_scores) / count
    average_i_bleu = sum(i_bleu_scores) / count
    average_rouge_l = sum(rouge_l_scores) / count

    print(f"Average BLEU score (first 100 samples): {average_bleu:.2f}")
    print(f"Average iBLEU score (first 100 samples): {average_i_bleu:.2f}")
    print(f"Average ROUGE-L score (first 100 samples): {average_rouge_l:.2f}")
