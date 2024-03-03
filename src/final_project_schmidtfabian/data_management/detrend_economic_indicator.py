import statsmodels.api as sm
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def detrend_economic_indicator(dataframe_economic_indicator):
    """
    This function detrends the economic activity indicator by using the Hodrick-Prescott filter. Since there do not exist any recommendations for the lambda value for the HP-filter, I use the Ravn & Uhlig formula and then adjust the value until the trend time-series looks close to the quarterly time-series.
    This is because there are recommendations for the lambda value for quarterly time-series data.
    Args:
    dataframe_economic_indicator(pd.Dataframe): A cleaned pandas dataframe containing the economic indicator.

    Returns:
    detrended_data(pd.Dataframe)
    """
    _fail_if_wrong_dataframe(dataframe=dataframe_economic_indicator)
    detrended_data = dataframe_economic_indicator.copy(deep = True)
    lambda_value=5.75*365**4
    detrended_data["cycle_values"], detrended_data["trend_values"] = sm.tsa.filters.hpfilter(detrended_data["values"].values, lamb=lambda_value)
    return detrended_data


def _fail_if_wrong_dataframe(dataframe):
    """Throws an error when argument is invalid."""
    _fail_if_not_pandas_dataframe(dataframe=dataframe)
    _fail_if_not_contain_column_values(dataframe=dataframe)
    _fail_if_wrong_datatype_column_values(dataframe=dataframe)
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
    
def _fail_if_not_contain_column_values(dataframe):
    """ Throws an error if the column 'values' is not contained in the dataframe."""
    if "values" not in dataframe.columns:
        raise ValueError("Column 'values' not found in the DataFrame.")
    
def _fail_if_wrong_datatype_column_values(dataframe):
    """Throws an error if the column 'values' in the argument does not have the correct data type."""
    actual_dtype = dataframe["values"].dtype
    if actual_dtype != ("float64" or "int") or actual_dtype == "bool":
        raise TypeError(f"Column 'values' has dtype {actual_dtype}, expected 'float64' or 'int'.")

def _fail_if_column_values_not_enough_elements(dataframe):
    """Throws an error if the column 'values' has less than two elements."""
    length_values_column = len(dataframe["values"])
    if length_values_column < 2:
        raise ValueError(f"Column 'values' needs to have at least two elements has {length_values_column}.")