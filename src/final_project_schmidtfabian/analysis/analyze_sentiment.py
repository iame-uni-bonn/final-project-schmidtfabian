import pandas as pd
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification, logging, TrainingArguments, Trainer
import torch

def analyze_sentiment(dataframe_headlines):
    dataframe_headlines_sentiment = dataframe_headlines.copy(deep = True)
    tokenizer = AutoTokenizer.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    model = AutoModelForSequenceClassification.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines",
                                                               from_tf=True)
    location_headlines="headlines in a list"
    dataframe_headlines_sentiment=_classify_sentiment(dataframe_headlines,model,tokenizer,location_headlines)
    return dataframe_headlines_sentiment
    


def _classify_sentiment(dataframe, model, tokenizer, column_to_analyze):
    sentiment_classifier = TextClassificationPipeline(tokenizer=tokenizer,
                                                      model=model,
                                                     device="cuda:0" if torch.cuda.is_available() else None)
    
    dataframe["sentiment"]=None
    dataframe["sentiment_per_element"]=None
    dataframe["sentiment_score_per_element"]=None

    for index in dataframe.index:
        result=sentiment_classifier(dataframe.loc[index, column_to_analyze])
        label_mapping = {'negative': 1, 'neutral': 2, 'positive': 3}
        label_integers = [label_mapping[item['label']] for item in result]
        label_scores = [item["score"] for item in result]
        mean_label = sum(label_integers) / len(label_integers)
        dataframe.loc[index,"sentiment"]=mean_label
        dataframe.at[index, "sentiment_per_element"] = label_integers 
        dataframe.at[index, "sentiment_score_per_element"]= label_scores
    return dataframe
    