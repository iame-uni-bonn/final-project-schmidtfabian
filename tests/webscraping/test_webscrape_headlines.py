import pytest
from datetime import datetime
import pandas as pd

from final_project_schmidtfabian.webscraping.webscrape_headlines import webscrape_headlines_welt

wrong_dtypes_starting_date = ["Hello", 5, 1.2, True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes_starting_date)
def test_webscrape_headlines_welt_wrong_dtypes_starting_date(input):
    with pytest.raises(TypeError):
        webscrape_headlines_welt(starting_date=input, number_of_days=100)

wrong_dtypes_number_of_days = ["Hello", 1.2, True, None, [1, 2, 3], {"a":1, "b":1}]

@pytest.mark.parametrize("input", wrong_dtypes_number_of_days)
def test_webscrape_headlines_welt_wrong_dtypes_number_of_days(input):
    with pytest.raises(TypeError):
        webscrape_headlines_welt(starting_date=datetime(2008,1,1), number_of_days=input)

def test_webscrape_headlines_welt_starting_date_before_2008():
    with pytest.raises(ValueError):
        webscrape_headlines_welt(starting_date=datetime(2007,12,31), number_of_days=100)

def test_webscrape_headlines_welt_number_of_days_out_of_reach():
    with pytest.raises(ValueError):
        webscrape_headlines_welt(starting_date=datetime(2024,1,1), number_of_days=1000)

def test_webscrape_headlines_welt_correct_index():
    test_dataframe_headlines = webscrape_headlines_welt(
        starting_date= datetime(2009,1,1),
        number_of_days=10
        )
    date_range = pd.date_range(start='2009-01-01', end='2009-01-10')
    assert test_dataframe_headlines.index.equals(date_range) and \
        test_dataframe_headlines.index.dtype == 'datetime64[ns]'