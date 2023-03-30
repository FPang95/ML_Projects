from summarizer import Summarizer, TransformerSummarizer
from rouge import Rouge
import pandas as pd

df = pd.read_csv("test.csv", dtype=str)

# Call bert pretrained summarizer for 100 articles
bert_model = Summarizer()
bert_summaries = [''.join(bert_model(x, num_sentences=3)) for x in df["articles"].iloc[:100]]

# get average Rouge scores
rouge = Rouge()
bert_scores = rouge.get_scores(bert_summaries, list(df["summary"].iloc[:100]), avg=True)


# Call GPT2 summarizer for 100 articles
gpt2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
gpt2_summaries = [''.join(gpt2_model(x, num_sentences=3)) for x in df["articles"].iloc[:100]]

gpt2_scores = rouge.get_scores(gpt2_summaries, list(df["summary"].iloc[:100]), avg=True)


# Call XLNet summarizer for 100 articles
xln_model = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")

xln_summaries = [''.join(xln_model(x, num_sentences=3)) for x in df["articles"].iloc[:100]]

xln_scores = rouge.get_scores(xln_summaries, list(df["summary"].iloc[:100]), avg=True)

print("BERT ROUGE")
print(bert_scores)

print("GPT2 ROUGE")
print(gpt2_scores)

print("XLNet ROUGE")
print(xln_scores)
