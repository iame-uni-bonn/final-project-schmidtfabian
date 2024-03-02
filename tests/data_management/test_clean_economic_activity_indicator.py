import pytest
import pandas as pd

from final_project_schmidtfabian.data_management.clean_economic_activity_indicator import clean_economic_activity_indicator

wrong_dtypes = [5, 1.2, "hello", True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes)
def test_clean_economic_indicator_wrong_dtypes(input):
    with pytest.raises(TypeError):
        clean_economic_activity_indicator(input)

test_dictionary_wrong_columns =  {"Datum gestern": ["25.02.2024", "26.02.2024", "27.02.2024"], "Kalender-, aber nicht saisonbereinigt (KNSB)":[100.5,99.7,95.4]}
test_dataframe_wrong_columns = pd.DataFrame(data = test_dictionary_wrong_columns)

def test_clean_economic_indicator_wrong_columns():
    with pytest.raises(ValueError):
        clean_economic_activity_indicator(test_dataframe_wrong_columns)


test_dictionary_correct_columns =  {"Datum": ["25.02.2024", "26.02.2024", "27.02.2024"], "Kalender- und saisonbereinigt (KSB)":[100.5,99.7,95.4]}
test_dataframe_correct_columns = pd.DataFrame(test_dictionary_correct_columns)

def test_clean_economic_indicator_correct_dtypes():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    assert isinstance(cleaned_data.index,pd.DatetimeIndex) and cleaned_data["values"].dtype == pd.Float64Dtype()

def test_clean_economic_indicator_rename():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    assert 'values' in cleaned_data.columns and 'Datum' not in cleaned_data.columns and 'Kalender- und saisonbereinigt (KSB)' not in cleaned_data.columns, \
    "Column 'values' is not in the DataFrame, or columns 'Datum' or 'Kalender- und saisonbereinigt (KSB)' exist in the DataFrame."

def test_clean_economic_indicator_correct_index():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    index_values_datetime = pd.to_datetime(["25.02.2024", "26.02.2024", "27.02.2024"], errors='coerce', dayfirst=True)
    assert cleaned_data.index.name == "date" and (cleaned_data.index == index_values_datetime).all(), \
        "Index values do not match the original values"

def test_clean_economic_indicator_correct_values():
    cleaned_data = clean_economic_activity_indicator(test_dataframe_correct_columns)
    assert (cleaned_data["values"].values==[100.5,99.7,95.4]).all(), \
        "The values in the column 'values' were not assigned correctly."