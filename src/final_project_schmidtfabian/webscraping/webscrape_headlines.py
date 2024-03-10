from requests_html import HTMLSession
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd


def webscrape_headlines_welt(starting_date, number_of_days):
    """Webscrape headlines from German newspaper 'Die Welt'.
    
    Webscrapes headlines from the German newspaper 'Die Welt' starting from the
    provided starting date until the number of days specified is reached.

    Args:
        - starting_date(datetime.datetime): The date at which the webscraping should
        start.
        - number_of_days(int): The number of days for which the headlines should be
        scraped.

    Returns:
        - headline_dataframe(pd.DataFrame)

    """
    _fail_invalid_inputs(starting_date=starting_date, number_of_days=number_of_days)
    date_list = _generate_date_list(start_date=starting_date,num_days=number_of_days)
    date_list_reformatted = _format_date_list(date_list=date_list)

    list_of_headlines = []
    headline_date = []

    session = HTMLSession()

    for date in date_list_reformatted:
        list_headlines = _collect_headlines_Welt(date=date, session=session)
        list_of_headlines.append(list_headlines)
        headline_date.append(date)
    
    data = {
        "headlines date": headline_date,
        "headlines in a list": list_of_headlines
    }
    headline_dataframe = pd.DataFrame(data)
    headline_dataframe['headlines date'] = pd.to_datetime(headline_dataframe['headlines date'], dayfirst=True)
    headline_dataframe.set_index('headlines date', inplace=True)
    return headline_dataframe


def _collect_headlines_Welt(date, session):
    """Collects headlines from website 'Die Welt' from the specified date."""
    url = f'https://www.welt.de/schlagzeilen/nachrichten-vom-{date}.html'
    response = session.get(url)

    soup = BeautifulSoup(response.html.html, 'html.parser')

    articles = soup.find_all('article')
    headlines = []
    category_subheadline_headline = []

    for article in articles:
        aria_label = article.find('a', class_='c-teaser__headline-link')['aria-label']
        category_subheadline_headline.append(aria_label)
    
    categories, headlines = zip(*[item.split(' - ', 1) if ' - ' in item else ['', item] for item in category_subheadline_headline])
    filtered_headlines = [headline for category, headline in zip(categories, headlines) if category in ['politik', 'wirtschaft', 'finanzen']]

    return filtered_headlines

def _generate_date_list(start_date, num_days):
    """Generates a list with a provided number of dates starting from starting date."""
    dates = []
    current_date = start_date
    for _ in range(num_days):
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def _format_date_list(date_list):
    """Formats date list into a list containing strings in the right format."""
    formatted_dates = []
    for date in date_list:
        formatted_date = date.strftime("%d-%m-%Y")
        formatted_dates.append(formatted_date)
    return formatted_dates

def _fail_invalid_inputs(starting_date, number_of_days):
    """Throws an error if inputs are invalid."""
    _fail_if_inputs_wrong_dtypes(starting_date=starting_date,number_of_days= number_of_days)
    _fail_if_number_of_days_out_of_reach(starting_date=starting_date, number_of_days=number_of_days)
    _fail_if_starting_date_is_before_2008(starting_date=starting_date)

def _fail_if_inputs_wrong_dtypes(starting_date, number_of_days):
    """Raises an error if inputs have wrong data types."""
    if not isinstance(starting_date, datetime) or not isinstance(number_of_days, int) or \
        isinstance(number_of_days, bool):
        raise TypeError("'starting_date' must be a datetime object and 'number_of_days' date must be an integer")
    
def _fail_if_number_of_days_out_of_reach(starting_date, number_of_days):
    """Raises an error if the number of days is not reachable."""
    current_date = datetime.now()
    last_date_headlines = starting_date + timedelta(days=number_of_days)
    if last_date_headlines > current_date:
        raise ValueError("The number of days are not reachable. Last date lies in the future.")

def _fail_if_starting_date_is_before_2008(starting_date):
    """Raises an error if the starting date is before 2008."""
    earliest_date = datetime(2008,1,1)
    if earliest_date > starting_date:
        raise ValueError("'starting date' must lie after the year 2007.")