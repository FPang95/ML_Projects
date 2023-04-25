from summarizer import Summarizer
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as ms
from bert_score import score
from nltk.tokenize import word_tokenize
import torch

df = pd.read_csv("test.csv", dtype=str)

# Call bert pretrained summarizer for 100 articles
bert_model = Summarizer('bert-base-uncased')
bert_summaries = [''.join(bert_model(x, num_sentences=3)) for x in df["articles"].iloc[:100]]

# get average Rouge scores
rouge = Rouge()
bert_scores = rouge.get_scores(bert_summaries, list(df["summary"].iloc[:100]), avg=True)

print("Rouge-1:", bert_scores["rouge-1"])
print("Rouge-2:", bert_scores["rouge-2"])
print("Rouge-L:", bert_scores["rouge-l"])

# Calculate BLEU score and print out average
bleu_scores = [sentence_bleu([word_tokenize(ref)], word_tokenize(gen)) for ref, gen in zip(df["summary"].iloc[:100], bert_summaries[:100])]
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU score:", average_bleu_score)

# Calculate METEOR scores for each pair of summaries and print out average
meteor_scores = [ms([word_tokenize(ref)], word_tokenize(gen)) for ref, gen in zip(list(df["summary"].iloc[:100]), bert_summaries)]
average_meteor_score = sum(meteor_scores) / len(meteor_scores)
print("Average METEOR Score:", average_meteor_score)

# Use a pre-trained BERT model ('bert-base-uncased')
P, R, F1 = score(bert_summaries, list(df["summary"].iloc[:100]), lang='en', model_type='bert-base-uncased')

average_P = torch.mean(P)
average_R = torch.mean(R)
average_F1 = torch.mean(F1)

print("Average BERTScore Precision:", average_P.item())
print("Average BERTScore Recall:", average_R.item())
print("Average BERTScore F1:", average_F1.item())
