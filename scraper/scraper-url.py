from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
import numpy as np
import sys

url_prefix = 'https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons?view=grid&page='

def scrape_urls(start_num, end_num):
    urls = []
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    # options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)

    for n in range(int(start_num), int(end_num) + 1):
        url_length = len(urls)

        url = url_prefix + str(n)
        driver.get(url)
        time.sleep(3) # Loading page might take time
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        result_container = soup.find_all('a', class_='animalCard--avL0R')
        urls.extend([a.get('href') for a in result_container if a.get('href')])
        assert len(urls) > url_length, 'Nothing is returned'

        print(f'page {n} complete')

        if n % 5 == 0:
            pd.DataFrame(urls, columns=['url']).to_csv('data/url_df.csv', index=False)
    
    driver.quit()
    return len(urls)

if __name__ == '__main__':
    args = sys.argv[1:]
    
    if len(args) == 2:
        scrape_urls(args[0], args[1])
