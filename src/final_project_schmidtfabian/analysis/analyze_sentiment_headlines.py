import pandas as pd
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification
import torch
import os

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def analyze_sentiment_zero_shot_classification(dataframe_headlines):
    """Analyzes the sentiment of headlines using zero-shot-classification.
    
    Analyzes the text that are contained in the column 'headlines in a list' using the multilingual-sentiment-
    newspaper-headlines model by Zachary Dickson. The three labels that can be assigned to a headline are
    0 for 'negative', 1 for 'neutral' and 2 for 'positive'. Adds a column containing the label for every
    individual headline as well as a mean of all headlines in the list and also adds the score for each
    individual headline in a list. The score is the probability assigned by the model that this is
    the correct label.

    Args:
        - dataframe_headlines(pd.Dataframe): The dataframe containing the headlines in the column
        'headlines in a list.

    Returns:
        - dataframe_headlines_sentiment
    
    """
    _fail_if_wrong_input(dataframe=dataframe_headlines)
    dataframe_headlines_sentiment = dataframe_headlines.copy(deep = True)
    tokenizer = AutoTokenizer.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    model = AutoModelForSequenceClassification.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines",
                                                               from_tf=True)
    location_headlines="headlines in a list"
    dataframe_headlines_sentiment=_classify_sentiment_zero_shot(dataframe=dataframe_headlines,
                                                                model=model,
                                                                tokenizer=tokenizer,
                                                                column_to_analyze=location_headlines)
    return dataframe_headlines_sentiment

def analyze_sentiment_finetuned_model(dataframe_headlines, location_finetuned_model):
    """Analyzes the sentiment of headlines using a fine-tuned model.
    
    Analyzes the text that are contained in the column 'headlines in a list' using the finetuned
    multilingual-sentiment-newspaper-headlines model by Zachary Dickson. The three labels that can be
    assigned to a headline are 0 for 'negative', 1 for 'neutral' and 2 for 'positive'. Adds a column
    containing the label for every individual headline as well as a mean of all headlines in the list
    and also adds the score for each individual headline in a list. The score is the probability assigned
    by the model that this is the correct label.

    Args:
        - dataframe_headlines(pd.Dataframe): The dataframe containing the headlines in the column
        'headlines in a list.
        - location_finetuned_model(Path): The Path where the fine-tuned model is located in.

    Returns:
        - dataframe_headlines_sentiment
    
    """
    _fail_if_wrong_input(dataframe=dataframe_headlines)
    _fail_if_location_finetuned_model_is_not_a_filepath(
        location_finetuned_model=location_finetuned_model
        )
    dataframe_headlines_sentiment = dataframe_headlines.copy(deep = True)
    tokenizer = AutoTokenizer.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    model = AutoModelForSequenceClassification.from_pretrained(location_finetuned_model)
    location_headlines="headlines in a list"
    dataframe_headlines_sentiment=_classify_sentiment_finetuned(dataframe=dataframe_headlines,
                                                                model=model,
                                                                tokenizer=tokenizer,
                                                                column_to_analyze=location_headlines)
    return dataframe_headlines_sentiment
    


def _classify_sentiment_zero_shot(dataframe, model, tokenizer, column_to_analyze):
    """Classifies sentiment of 'headlines in a list' using zero-shot-classification"""
    sentiment_classifier = TextClassificationPipeline(tokenizer=tokenizer,
                                                      model=model,
                                                     device="cuda:0" if torch.cuda.is_available() else None)
    
    dataframe_sentiment_analyzed = dataframe.copy(deep = True)
    dataframe_sentiment_analyzed = _create_sentiment_columns(dataframe=dataframe_sentiment_analyzed)

    for index in dataframe_sentiment_analyzed.index:
        result=sentiment_classifier(dataframe_sentiment_analyzed.loc[index, column_to_analyze].tolist())
        dataframe_sentiment_analyzed = _add_results_to_dataframe(dataframe=dataframe_sentiment_analyzed,result=result, index=index)
    
    dataframe_sentiment_analyzed['sentiment_score_per_element'] = dataframe_sentiment_analyzed['sentiment_score_per_element'].apply(_round_scores)
    dataframe_sentiment_analyzed["sentiment"] = dataframe_sentiment_analyzed["sentiment"].astype(
        pd.Float64Dtype()
    )

    return dataframe_sentiment_analyzed

def _classify_sentiment_finetuned(dataframe, model, tokenizer, column_to_analyze):
    """Classifies sentiment of 'headlines in a list' using fine-tuned model."""
    sentiment_classifier = TextClassificationPipeline(tokenizer=tokenizer,
                                                      model=model,
                                                     device="cuda:0" if torch.cuda.is_available() else None,
                                                     framework="pt")
    
    dataframe_sentiment_analyzed = dataframe.copy(deep = True)
    dataframe_sentiment_analyzed = _create_sentiment_columns(dataframe=dataframe_sentiment_analyzed)

    for index in dataframe_sentiment_analyzed.index:
        result=sentiment_classifier(dataframe_sentiment_analyzed.loc[index, column_to_analyze].tolist())
        dataframe_sentiment_analyzed = _add_results_to_dataframe(dataframe=dataframe_sentiment_analyzed,result=result, index=index)
    
    dataframe_sentiment_analyzed['sentiment_score_per_element'] = dataframe_sentiment_analyzed['sentiment_score_per_element'].apply(_round_scores)
    dataframe_sentiment_analyzed["sentiment"] = dataframe_sentiment_analyzed["sentiment"].astype(
        pd.Float64Dtype()
    )

    return dataframe_sentiment_analyzed


def _add_results_to_dataframe(dataframe, result, index):
    """Adds result of sentiment classifier to the provided index of the dataframe."""
    label_mapping = {'negative': 1, 'neutral': 2, 'positive': 3}
    label_integers = [label_mapping[item['label']] for item in result]
    label_scores = [item["score"] for item in result]
    mean_label = sum(label_integers) / len(label_integers)
    dataframe.loc[index,"sentiment"]=mean_label
    dataframe.at[index, "sentiment_per_element"] = label_integers 
    dataframe.at[index, "sentiment_score_per_element"]= label_scores
    return dataframe

def _create_sentiment_columns(dataframe):
    """Creates three new columns storing the values of the sentiment analyzing."""
    dataframe["sentiment"]=None
    dataframe["sentiment_per_element"]=None
    dataframe["sentiment_score_per_element"]=None
    return dataframe

def _round_scores(score_list):
    """Rounds the scores in a list to two digits after the comma."""
    decimal_places = 2
    return [round(score, decimal_places) for score in score_list]

def _fail_if_wrong_input(dataframe):
    """Throws an error if the input is invalid."""
    _fail_if_not_dataframe(dataframe=dataframe)
    _fail_does_not_contain_column_headlines_in_a_list(dataframe=dataframe)
    _fail_if_elements_in_column_headlines_in_a_list_have_wrong_dtype(dataframe=dataframe)

def _fail_if_not_dataframe(dataframe):
    """Throws an error if the input is not a pandas dataframe."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Input is not a dataframe, it is {type(dataframe)}.")
    
def _fail_does_not_contain_column_headlines_in_a_list(dataframe):
    """Throws an error when 'headlines in a list' column is not in dataframe."""
    if "headlines in a list" not in dataframe.columns:
        raise ValueError("Column 'headlines in a list' not found in the dataframe.")
    
def _fail_if_elements_in_column_headlines_in_a_list_have_wrong_dtype(dataframe):
    """Throws an error if elements in 'headlines in a list' has wrong data type."""
    headlines_list = dataframe.iloc[0]["headlines in a list"]
    column_dtype = type(headlines_list[0])
    if not isinstance(headlines_list[0], str):
        raise ValueError(f"Column 'headlines in a list' is not a list containing \
                        strings , its element's datatype is {column_dtype}.")

def _fail_if_location_finetuned_model_is_not_a_filepath(
        location_finetuned_model
        ):
    if not os.path.isabs(location_finetuned_model):
        raise TypeError("'location_finetuned_model' has to be a valid absolute path.")