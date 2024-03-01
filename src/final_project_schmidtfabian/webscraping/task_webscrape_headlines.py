import pytask
import pandas as pd
from pathlib import Path
from datetime import datetime

from final_project_schmidtfabian.config import BLD
from final_project_schmidtfabian.webscraping.webscrape_headlines import webscrape_headlines_welt


webscrape_headlines_deps = {
    "scripts": Path("webscrape_headlines.py")
}

def task_webscrape_headlines(depends_on = webscrape_headlines_deps,
                             produces = BLD / "data" / "dataset_headlines.arrow"):
    starting_date = datetime(2008,1,1)
    dataframe_headlines = webscrape_headlines_welt(starting_date=starting_date, number_of_days=1000)
    dataframe_headlines.to_feather(produces)
