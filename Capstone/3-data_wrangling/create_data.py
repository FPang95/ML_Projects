from datasets import load_dataset
import pandas as pd


def create_dataset_csv(dataset_type):
    # Creates a two column csv of news articles and summaries
    # for text summarization ML

    # Load data (list of dictionaries)
    dataset = load_dataset("cnn_dailymail", '3.0.0')

    # create new dataframe and reformat data into
    # article and summary
    new_dataset = pd.DataFrame()

    article = [x['article'] for x in dataset[dataset_type]]
    summary = [x['highlights'] for x in dataset[dataset_type]]

    new_dataset["articles"] = article
    new_dataset["summary"] = summary

    # save data as CSV
    new_dataset.to_csv("{}.csv".format(dataset_type), index=False)


create_dataset_csv('test')
create_dataset_csv('train')
create_dataset_csv('validation')
