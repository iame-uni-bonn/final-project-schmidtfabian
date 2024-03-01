import pandas as pd
from datasets import Dataset, DatasetDict

def create_datasetdict(dataframe):
    # Splitting dataframe into test and validation set
    train_df = dataframe.sample(frac=0.666666666, random_state=5)
    validation_df = dataframe.drop(train_df.index)
    test_df = validation_df.sample(frac=0.5, random_state=56)
    validation_df = validation_df.drop(test_df.index)

    # Create a dictionary with the train and validation splits
    data_dict = {
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(validation_df),
        "test": Dataset.from_pandas(test_df)
    }

    
    data_dict = _remove_wrongly_created_index_column(data_dict=data_dict)

    # Convert to a DatasetDict
    dataset_dict_headlines = DatasetDict(data_dict)
    return dataset_dict_headlines

def _remove_wrongly_created_index_column(data_dict):
    for part_from_dataset in ["train","validation","test"]:
        if "__index_level_0__" in data_dict[part_from_dataset].column_names:
            data_dict[part_from_dataset] = data_dict[part_from_dataset].remove_columns("__index_level_0__")
        
    return data_dict