from flask import Flask, request, render_template
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Summarizer:
    def __init__(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    def generate_summary(self, text, max_length=128):
        input_ids = self.tokenizer.encode("summarize: " + text, return_tensors="pt")
        summary_ids = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


app = Flask(__name__)

model_path = os.getenv('MODEL_PATH', "./finetuned_t5_v2")
summarizer = Summarizer(model_path)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/summarize', methods=['POST'])
def summarize_route():
    article = request.form.get('article')
    summary = summarizer.generate_summary(article)
    logger.info(f"Generated summary: {summary}")
    return render_template('result.html', summary=summary)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
