Article Summarizer Model

Several LLMs were evaluated to be trained for a text summarizer model including BERT, BART, GTP2 and T5 (2-model_benchmark, 4-existing_paper, 5-experimenting_with_models). Several factors were taken into consideration including ease of training and performance on several text summarization metrics (5-experimenting_with_models). The T5 model was ultimately chosen because it was already pre-trained on the task of text summarization, performing the best across the evaluation metrics, and also allowed for fine-tuning.

Data was collected from the datasets library from HuggingFace and included several hundred thousand news articles along with their summaries from CNN and DailyMail.

One of the most significant challenges of finetuning the model was the training step due to the large amount of data resulting in the need for a powerful computational resource. The training was done by using a GPU-compatible Pytorch (with an RTX 3070 for nearly 2 hours) on a set of 100,000 training and 20,000 validation articles (6-ml_prototype). The finetuned T5 model was subsequently evaluated on a set of 250 new articles (7-scale_prototype).

The Docker image consists of the fine-tuned T5 model (includes the Pytorch .bin model, and configurations file), the requirements.txt, the Dockerfile, the HTML files for the flask application, and the script to run the flask application (8-deploy_to_docker).

The docker container was pushed to the Google Container Registry (GCR) and the image was deployed on Google Cloud Run

The final model can be accessed here: https://text-summarizer-service-gdt66y45aq-nn.a.run.app
