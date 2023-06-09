import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, Trainer, TrainingArguments, EarlyStoppingCallback
import pandas as pd

# load in training and validation data
df_train = pd.read_csv("train.csv", dtype=str)
df_valid = pd.read_csv("validation.csv", dtype=str)

# take subset for model fine-tuning
data_train = df_train[:100000]
data_valid = df_valid[:20000]

# create class to tokenize data


class SummaryDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length, target_max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data.iloc[idx]["articles"]
        target = self.data.iloc[idx]["summary"]

        source_tokenized = self.tokenizer(
            source,
            max_length=self.source_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_tokenized = self.tokenizer(
            target,
            max_length=self.target_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_tokenized["input_ids"].squeeze(),
            "attention_mask": source_tokenized["attention_mask"].squeeze(),
            "labels": target_tokenized["input_ids"].squeeze(),
        }


# set variables for trainer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

# set input lengths for training and validation data
source_max_length = 512
target_max_length = 128

# Tokenize training and validation data
train_dataset = SummaryDataset(data_train, tokenizer, source_max_length, target_max_length)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

valid_dataset = SummaryDataset(data_valid, tokenizer, source_max_length, target_max_length)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

# set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    # resume_from_checkpoint="./results/checkpoint-700"
)

# create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Run trainer
print(f"Resuming training from {training_args.resume_from_checkpoint}")
trainer.train()

# save model and tokenizer
model.save_pretrained("./finetuned_t5")
tokenizer.save_pretrained("./finetuned_t5")
