from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification, \
    logging, TrainingArguments, Trainer
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score

def create_trainer_for_text_classification_model(saving_location_model,dataset_dict_headlines):
    """
    Creates a trainer instance that is then used to train the multilingual_sentiment_newspaper_headlines
    by z-dickson. Specifies Arguments that load the best model at the end and set seeds to ensure 
    reproducibility.
    
    Args:  
        - saving_location_model(Path): This is the path to the directory where the model should be saved at.
        - dataset_dict_headlines(DatasetDict): The dataset dictionary containing a train, and validation
        dataset used for training the model.
        
    Returns:
        - trainer(transformers.Trainer)
    
    """
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
    """Computes macro f1-score and accuracy score using the predictions of the model during the
    training process."""
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}