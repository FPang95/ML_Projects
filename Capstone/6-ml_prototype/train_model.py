import torch
from transformers import BertTokenizerFast, EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd

# Read in training and validation data
df_train = pd.read_csv("train.csv", dtype=str)
df_val = pd.read_csv("validation.csv", dtype=str)

# define tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Create tokenizer function
def tokenize_data(example):
    inputs = tokenizer(example["articles"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = outputs["input_ids"]
    return inputs


train_dataset = df_train.apply(lambda x: tokenize_data(x), axis=1)
valid_dataset = df_val.apply(lambda x: tokenize_data(x), axis=1)

# Load pretrained model
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Set the decoder_start_token_id and pad_token_id attributes
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Set up the training arguments and trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
