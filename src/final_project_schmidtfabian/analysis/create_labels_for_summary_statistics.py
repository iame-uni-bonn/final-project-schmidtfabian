from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.config import BLD

def create_labels_for_summary_statistics(test_data_datasetdict_headlines):
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