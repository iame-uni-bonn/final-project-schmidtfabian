import pandas as pd
from datasets import Dataset, DatasetDict

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def create_datasetdict(dataframe):
    """
    Creates a dataset dictionary containing a train, test and validation dataset.

    Args:
    dataframe(pandas.Dataframe): A pandas dataframe containing the elements that should be seperated.

    Returns:
    dataset_dict_headlines(DatasetDict)
    """
    _fail_if_invalid_input(dataframe=dataframe)
    train_df = dataframe.sample(frac=0.666666666, random_state=5)
    validation_df = dataframe.drop(train_df.index)
    test_df = validation_df.sample(frac=0.5, random_state=56)
    validation_df = validation_df.drop(test_df.index)

    data_dict = {
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(validation_df),
        "test": Dataset.from_pandas(test_df)
    }

    data_dict = _remove_wrongly_created_index_column(data_dict=data_dict)
    dataset_dict_headlines = DatasetDict(data_dict)

    return dataset_dict_headlines

def _remove_wrongly_created_index_column(data_dict):
    """Removes the falsely created index column."""
    for part_from_dataset in ["train","validation","test"]:
        if "__index_level_0__" in data_dict[part_from_dataset].column_names:
            data_dict[part_from_dataset] = data_dict[part_from_dataset].remove_columns("__index_level_0__")
        
    return data_dict

def _fail_if_invalid_input(dataframe):
    """Throws an error if argument is invalid."""
    _fail_if_not_pandas_dataframe(dataframe=dataframe)
    _fail_if_column_values_not_enough_elements(dataframe=dataframe)

def _fail_if_not_pandas_dataframe(dataframe):
    """Throws an error if argument is not a pandas Dataframe."""
    if not isinstance(dataframe,pd.DataFrame):
        current_datatype = type(dataframe)
        msg = (
            f"{dataframe} has to be a pandas dataframe, is currently: {current_datatype}."
        )
        raise TypeError(
            msg,
        )

def _fail_if_column_values_not_enough_elements(dataframe):
    """Throws an error if the column 'values' has less than two elements."""
    length_values_column = len(dataframe.index)
    if length_values_column < 5:
        raise ValueError(f"Column 'values' needs to have at least five elements has {length_values_column}.")