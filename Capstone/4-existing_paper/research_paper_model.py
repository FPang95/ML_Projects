"""
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
Generation, Translation, and Comprehension
https://arxiv.org/pdf/1910.13461.pdf

Running code
from https://huggingface.co/facebook/bart-large-cnn

"""

from transformers import pipeline
from rouge import Rouge
import pandas as pd

# read in training dataset
train_df = pd.read_csv("test.csv",dtype=str)

# instantiate BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# get summaries for first 10 summaries
bart_summaries = [summarizer(x, min_length=30, do_sample=False)[0]["summary_text"]
                  for x in train_df["articles"].iloc[:10]]

# get average Rouge score
rouge = Rouge()
bart_scores = rouge.get_scores(bart_summaries,list(train_df["summary"].iloc[:10]),avg=True)

print("BART ROUGE")
print(bart_scores)
# ROUGE1: precision - 0.359, recall - 0.355, F1 score - 0.352
# ROUGE2: precision - 0.173, recall - 0.181, F1 score - 0.174
# ROUGEL: precision - 0.337, recall - 0.334, F1 score - 0.331
