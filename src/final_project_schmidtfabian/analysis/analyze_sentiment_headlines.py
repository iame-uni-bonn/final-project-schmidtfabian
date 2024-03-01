import pandas as pd
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification, logging, TrainingArguments, Trainer
import torch

def analyze_sentiment_zero_classification(dataframe_headlines):
    dataframe_headlines_sentiment = dataframe_headlines.copy(deep = True)
    tokenizer = AutoTokenizer.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    model = AutoModelForSequenceClassification.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines",
                                                               from_tf=True)
    location_headlines="headlines in a list"
    dataframe_headlines_sentiment=_classify_sentiment(dataframe_headlines,model,tokenizer,location_headlines)
    return dataframe_headlines_sentiment

def analyze_sentiment_finetuned_model(dataframe_headlines):
    dataframe_headlines_sentiment = dataframe_headlines.copy(deep = True)
    tokenizer = AutoTokenizer.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    model = AutoModelForSequenceClassification.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines",
                                                               from_tf=True)
    location_headlines="headlines in a list"
    dataframe_headlines_sentiment=_classify_sentiment(dataframe_headlines,model,tokenizer,location_headlines)
    return dataframe_headlines_sentiment
    


def _classify_sentiment(dataframe, model, tokenizer, column_to_analyze):
    """Classifies the sentiment of a specified column of a dataframe into three categories: 'positive', 'negative' and 'neutral' using the specified model and tokenizer."""
    sentiment_classifier = TextClassificationPipeline(tokenizer=tokenizer,
                                                      model=model,
                                                     device="cuda:0" if torch.cuda.is_available() else None)
    
    dataframe_sentiment_analyzed = dataframe.copy(deep = True)
    dataframe_sentiment_analyzed = _create_sentiment_columns(dataframe=dataframe_sentiment_analyzed)

    for index in dataframe_sentiment_analyzed.index:
        result=sentiment_classifier(dataframe_sentiment_analyzed.loc[index, column_to_analyze])
        dataframe_sentiment_analyzed = _add_results_to_dataframe(dataframe=dataframe_sentiment_analyzed,result=result, index=index)
    
    dataframe_sentiment_analyzed['sentiment_score_per_element'] = dataframe_sentiment_analyzed['sentiment_score_per_element'].apply(_round_scores)

    return dataframe_sentiment_analyzed


def _add_results_to_dataframe(dataframe, result, index):
    """Adds the result of the classification of the sentiment classifier to the provided index of the provided dataframe."""
    label_mapping = {'negative': 1, 'neutral': 2, 'positive': 3}
    label_integers = [label_mapping[item['label']] for item in result]
    label_scores = [item["score"] for item in result]
    mean_label = sum(label_integers) / len(label_integers)
    dataframe.loc[index,"sentiment"]=mean_label
    dataframe.at[index, "sentiment_per_element"] = label_integers 
    dataframe.at[index, "sentiment_score_per_element"]= label_scores
    return dataframe

def _create_sentiment_columns(dataframe):
    dataframe["sentiment"]=None
    dataframe["sentiment_per_element"]=None
    dataframe["sentiment_score_per_element"]=None
    return dataframe

def _round_scores(score_list):
    """Rounds the scores in a list to two digits after the comma."""
    decimal_places = 2
    return [round(score, decimal_places) for score in score_list]