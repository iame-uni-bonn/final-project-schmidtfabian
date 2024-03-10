from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification, \
    logging, TrainingArguments, Trainer
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import os

def create_trainer_for_text_classification_model(saving_location_model,dataset_dict_headlines):
    """Creates trainer for text-classification model.

    Creates a trainer instance that is then used to train the
    multilingual_sentiment_newspaper_headlines model by z-dickson. Specifies Arguments
    that load the best model at the end and set seeds to ensure reproducibility.
    
    Args:  
        - saving_location_model(Path): This is the path to the directory where the model should be saved at.
        - dataset_dict_headlines(DatasetDict): The dataset dictionary containing a train, and validation
        dataset used for training the model.
        
    Returns:
        - trainer(transformers.Trainer)
    
    """
    _fail_invalid_inputs(saving_location=saving_location_model,
                         datasetdict=dataset_dict_headlines)
    dataset_dict_encoded = _tokenize_data_and_set_right_format(data=dataset_dict_headlines)
    batch_size = 50
    logging_steps = len(dataset_dict_encoded["train"]) // batch_size
    model_name="z-dickson/multilingual_sentiment_newspaper_headlines"
    num_labels = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, from_tf=True
    ).to(device)
    training_args = TrainingArguments(
        output_dir=saving_location_model,
        overwrite_output_dir=True,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        seed= 451,
        data_seed= 892
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=dataset_dict_encoded["train"],
        eval_dataset=dataset_dict_encoded["validation"],
    )
    return trainer


def _tokenize_data_and_set_right_format(data):
    """Tokenizes the entire dataset and then sets the format to torch for training."""
    data_encoded = data.map(_tokenize, batched=True, batch_size=None)
    data_encoded.set_format("torch")
    return data_encoded



def _tokenize(batch):
    """Tokenizes a batch of text using the specified tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("z-dickson/multilingual_sentiment_newspaper_headlines")
    return tokenizer(batch["text"], padding=True, truncation=True)

def _compute_metrics(pred):
    """Computes macro f1-score and accuracy score using model's predictions."""
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def _fail_invalid_inputs(saving_location, datasetdict):
    _fail_if_saving_location_finetuned_model_is_not_a_filepath(saving_location)
    _fail_if_wrong_dtype_datasetdict(datasetdict)
    _fail_if_no_dataset_train_and_validation(dataset_dict=datasetdict)
    _fail_if_wrong_input_dataset(datasetdict["train"])
    _fail_if_wrong_input_dataset(datasetdict["validation"])


def _fail_if_saving_location_finetuned_model_is_not_a_filepath(
        location_finetuned_model
        ):
    if not os.path.isabs(location_finetuned_model):
        raise TypeError("'location_finetuned_model' has to be a valid absolute path.")
    
def _fail_if_wrong_dtype_datasetdict(dataset_dict):
    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(f"'dataset_dict_headlines' has to be a DatasetDict \
                        object, is currently '{type(dataset_dict)}'.")

def _fail_if_no_dataset_train_and_validation(dataset_dict):
    if "train" not in dataset_dict or "validation" not in dataset_dict:
        missing_datasets = [dataset_name for dataset_name in ["train", "validation"] if dataset_name not in dataset_dict]
        raise ValueError(f"DatasetDict is missing the following datasets: {', '.join(missing_datasets)}.")

def _fail_if_wrong_input_dataset(dataset):
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