import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

def clean_economic_activity_indicator(economic_indicator_dataframe):
    """
    Cleans the dataframe containing the economic activity indicator used in the rest of the project.
    Converts the columns into the right datatypes, renames columns and sets the right index.

    Args:
    economic_indicator_dataframe(pd.Dataframe): A pandas dataframe containing the economic activity indicator.

    Returns:
    cleaned_data(pd.Dataframe)
    """
    _fail_if_wrong_dataframe(economic_indicator_dataframe)
    cleaned_data = economic_indicator_dataframe.copy(deep = True)
    cleaned_data["Datum"] = pd.to_datetime(cleaned_data["Datum"], errors="raise", dayfirst=True)
    cleaned_data.rename(columns={"Datum": 'date'}, inplace=True)
    cleaned_data.rename(columns={"Kalender- und saisonbereinigt (KSB)": "values"}, inplace=True)
    cleaned_data["values"]=pd.to_numeric(cleaned_data["values"], errors= "raise")
    cleaned_data["values"] = cleaned_data["values"].astype(pd.Float64Dtype())
    cleaned_data.set_index('date', inplace=True)
    return cleaned_data


def _fail_if_wrong_dataframe(dataframe):
    """Throws an error if argument is not a pandas dataframe containing the correct columns."""
    _fail_if_not_pandas_dataframe(dataframe=dataframe)
    _fail_if_not_contains_columns(dataframe=dataframe)



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

def _fail_if_not_contains_columns(dataframe):
    """Throws an error if the columns 'Datum' and 'Kalender- und saisonbereinigt (KSB)' are not contained in the dataframe."""
    missing_columns = [col for col in ["Datum", "Kalender- und saisonbereinigt (KSB)"] if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Columns {', '.join(missing_columns)} not found in the DataFrame.")
