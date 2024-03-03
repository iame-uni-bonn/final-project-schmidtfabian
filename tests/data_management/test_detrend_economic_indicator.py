import pytest
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

from final_project_schmidtfabian.data_management.detrend_economic_indicator import detrend_economic_indicator

WRONG_DTYPES = [5, 1.2, "hello", True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", WRONG_DTYPES)
def test_detrend_economic_indicator_wrong_dtypes(input):
    """Tests if function raises a TypeError for inputs with the wrong datatype."""
    with pytest.raises(TypeError):
        detrend_economic_indicator(input)

test_dictionary_wrong_columns =  {"Datum": ["25.02.2024", "26.02.2024", "27.02.2024"],
                                  "Kalender- und saisonbereinigt (KSB)":[100.5,99.7,95.4]}
test_dataframe_wrong_columns = pd.DataFrame(test_dictionary_wrong_columns)

def test_detrend_economic_indicator_wrong_columns():
    """Test if function raises a ValueError for inputs with the wrong column names."""
    with pytest.raises(ValueError):
        detrend_economic_indicator(test_dataframe_wrong_columns)

test_dictionary_columns_wrong_dtypes =  {"date": ["25.02.2024", "26.02.2024", "27.02.2024"],
                                  "values":[True, False, True]}
test_dataframe_columns_wrong_dtypes = pd.DataFrame(test_dictionary_columns_wrong_dtypes)

def test_detrend_economic_indicator_columns_wrong_dtypes():
    """Tests if function raises a TypeError for  inputs where the column 'values' has the wrong data type."""
    with pytest.raises(TypeError):
        detrend_economic_indicator(test_dataframe_columns_wrong_dtypes)

test_dictionary_no_deviations_from_trend =  {"date": ["25.02.2024", "26.02.2024", "27.02.2024"],
                                  "values":[100.0,100.0,100.0]}
test_dataframe_no_deviations_from_trend = pd.DataFrame(test_dictionary_no_deviations_from_trend)

def test_detrend_economic_indicator_no_deviations():
    """Tests if trend values are all equal to actual values for no cyclical deviations."""
    no_deviations_dataframe = detrend_economic_indicator(test_dataframe_no_deviations_from_trend)
    assert np.all(np.isclose(no_deviations_dataframe["trend_values"], 100.0, atol=0.01)) \
       and np.all(np.isclose(no_deviations_dataframe["cycle_values"], 0.0, atol=0.01)), \
       "Trend values should be approximately 100.0 and cycle values should be approximately 0."

def test_detrend_economic_indicator_not_enough_elements():
    """Tests if function raises an error if the number of elements in the column 'values' is too small."""
    with pytest.raises(ValueError):
        detrend_economic_indicator(pd.DataFrame({"values":[100.0]}))

