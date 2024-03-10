from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from datasets import Dataset
import torch

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD

def create_labels_for_summary_statistics(test_data_datasetdict_headlines):
    """Create labels for summary statistics from the training process.
    
    Creates a dataframe containing three columns: one containing the hand-labelled
    sentiment, one containing the sentiment assigned by the fine-tuned model and one
    containing the sentiment assigned by the zero-shot classification model.

    Args:
        - test_data_datsetdict_headlines(datasets.DataSet): A dataset containing the
        headlines in the column 'text' and the hand-labelled labels in the column
        'label'.

    Returns:
        - dataframe_labels_test_data(pd.DataFrame)

    """
    _fail_if_wrong_input(dataset=test_data_datasetdict_headlines)
    tokenizer = AutoTokenizer\
        .from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    model_zero_shot_classification = AutoModelForSequenceClassification\
        .from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines",
                                                               from_tf=True)
    sentiment_classifier_zero_shot_classification = TextClassificationPipeline(
        tokenizer=tokenizer,
        model=model_zero_shot_classification,
        device="cuda:0" if torch.cuda.is_available() else None
        )
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(
        BLD / "finetuned_text_classification_model" / "checkpoint-4"
        )
    sentiment_classifier_finetuned_model = TextClassificationPipeline(
        tokenizer=tokenizer,
        model=model_finetuned,
        device="cuda:0" if torch.cuda.is_available() else None,
        framework="pt"
        )
    sentiment_labels_finetuned_model = pd.DataFrame(
        sentiment_classifier_finetuned_model(test_data_datasetdict_headlines["text"])
        )
    label_sentiment_to_int = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    sentiment_labels_int_finetuned_model = sentiment_labels_finetuned_model["label"].map(
        label_sentiment_to_int
        )
    sentiment_labels_zero_shot_classification = pd.DataFrame(
        sentiment_classifier_zero_shot_classification(test_data_datasetdict_headlines["text"])
        )
    sentiment_labels_int_zero_shot_classification =\
        sentiment_labels_zero_shot_classification["label"].map(
        label_sentiment_to_int
        )
    dataframe_labels_test_data = pd.DataFrame()
    dataframe_labels_test_data["label"] = test_data_datasetdict_headlines["label"]
    dataframe_labels_test_data["label finetuned model"] =\
    sentiment_labels_int_finetuned_model 
    dataframe_labels_test_data["label zero_shot_classification model"] =\
    sentiment_labels_int_zero_shot_classification
    return dataframe_labels_test_data

def _fail_if_wrong_input(dataset):
    _fail_if_wrong_dtype(dataset=dataset)
    _fail_if_wrong_columns(dataset=dataset)
    _fail_if_column_text_is_not_a_string(dataset=dataset)
    _fail_if_column_label_is_not_int(dataset=dataset)

def _fail_if_wrong_dtype(dataset):
    if not isinstance(dataset, Dataset):
        raise TypeError(f"'test_data_datasetdict_headlines' has to be a DataSet \
                        object, is currently '{type(dataset)}'.")
    
def _fail_if_wrong_columns(dataset):
    missing_columns = [col for col in ["text", "label"] if col not\
                        in dataset.column_names]
    if missing_columns:
        raise ValueError(f"Column(s) '{missing_columns}' not found in the dataset.")
    
def _fail_if_column_text_is_not_a_string(dataset):
    column_dtype = dataset.features["text"].dtype
    if column_dtype != 'large_string':
        raise ValueError(f"Column 'text' does not have datatype string,\
                          its datatype is {column_dtype}.")

def _fail_if_column_label_is_not_int(dataset):
    column_dtype = dataset.features["label"].dtype
    if column_dtype != 'int64':
        raise ValueError(f"Column 'label' does not have datatype 'int64',\
                          its datatype is {column_dtype}.")