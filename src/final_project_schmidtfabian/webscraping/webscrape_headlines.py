from requests_html import HTMLSession
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd


def webscrape_headlines_welt(starting_date, number_of_days):
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
    dates = []
    current_date = start_date
    for _ in range(num_days):
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def _format_date_list(date_list):
    formatted_dates = []
    for date in date_list:
        formatted_date = date.strftime("%d-%m-%Y")
        formatted_dates.append(formatted_date)
    return formatted_dates