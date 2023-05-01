from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as ms
from bert_score import score
from nltk.tokenize import word_tokenize
import torch

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate summaries


def generate_summary(text, model, tokenizer, max_length=128):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt")
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# read in test data
df = pd.read_csv("test.csv", dtype=str)

# benchmark using the first 250 articles
df = df[:250]

# generate the summaries
df["generated_summary"] = df["articles"].apply(lambda x: generate_summary(x, model, tokenizer))

# get average Rouge scores
rouge = Rouge()
t5_scores = rouge.get_scores(list(df["generated_summary"]), list(df["summary"]), avg=True)

print("Rouge-1:", t5_scores["rouge-1"])
print("Rouge-2:", t5_scores["rouge-2"])
print("Rouge-L:", t5_scores["rouge-l"])

# Calculate BLEU score and print out average
smooth_func = SmoothingFunction().method4
bleu_scores = [sentence_bleu([word_tokenize(ref)], word_tokenize(gen), smoothing_function=smooth_func) for ref, gen in zip(df["summary"], df["generated_summary"])]
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU score:", average_bleu_score)

# Calculate METEOR scores for each pair of summaries and print out average
meteor_scores = [ms([word_tokenize(ref)], word_tokenize(gen)) for ref, gen in zip(list(df["summary"]), df["generated_summary"])]
average_meteor_score = sum(meteor_scores) / len(meteor_scores)
print("Average METEOR Score:", average_meteor_score)

# Use a pre-trained BERT model ('bert-base-uncased')
P, R, F1 = score(list(df["generated_summary"]), list(df["summary"]), lang='en', model_type='bert-base-uncased')

average_P = torch.mean(P)
average_R = torch.mean(R)
average_F1 = torch.mean(F1)

print("Average BERTScore Precision:", average_P.item())
print("Average BERTScore Recall:", average_R.item())
print("Average BERTScore F1:", average_F1.item())
