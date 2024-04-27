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
import urllib
import sys
import gcsfs

complete_url = pd.read_csv('data/complete_url.csv')

def get_gene(soup):
    # Check if the container exists
    container = soup.find('div', class_='container--N6XSR')
    if not container:
        return []  # or return ['unknown'] if you prefer to indicate no genes found

    gene_classes = {
        'dom-codom': 'labelBadge--oeqO_ dom-codom--KcmpO',
        'het-rec': 'labelBadge--oeqO_ het-rec--Z8INA',
        'vis-rec': 'labelBadge--oeqO_ vis-rec--YbPYK',
        'pos-rec': 'labelBadge--oeqO_ pos-rec--TteXe',
        'polygenic': 'labelBadge--oeqO_ polygenic--GxLqN',
        'super-dom-codom': 'labelBadge--oeqO_ super-dom-codom--iEgCD'
    }
    
    # Gather all relevant gene traits
    gene_data = []
    for gene_type, class_name in gene_classes.items():
        # Find all span elements for each gene type based on its class name
        genes = {trait.get_text() for trait in container.find_all('span', class_=class_name)}
        gene_data.extend(genes)

    return gene_data


def get_origin(soup):
    classes_to_check = [
        'labelBadge--oeqO_ warning--WSnpA',
        'labelBadge--oeqO_ success--vMoO7',
        'labelBadge--oeqO_ danger--aAl4n'
    ]
    
    for class_name in classes_to_check:
        origin = [trait.get_text() for trait in soup.find_all('span', class_=class_name)]
        if origin:
            return origin[0]
        
    return 'unknown'


def get_sex(soup):
    if soup.find('svg', class_='svg-inline--fa fa-mars sex male'):
        return 'male'
    
    if soup.find('svg', class_='svg-inline--fa fa-venus sex female'):
        return 'female'
    
    if soup.find('svg', class_='svg-inline--fa fa-venus-mars sex mixed'):
        return 'mixed'
    
    return 'unknown'


def get_price(soup):
    try:
        price = float(soup.find_all('h1', class_='salePrice--qNIIs')[0].get_text().replace("$", "").replace(",", ""))
    except:
        price = 0
    return price


def get_birth(soup):
    try:
        birth_divs = [div for div in soup.find_all('div', class_='labelValueContainer--z1CP3')
                if div.find('b', string='Birth:')]
        birth_dates = [div.find('span').text for div in birth_divs]
        birth = birth_dates[0]
    except:
        birth = ''
    return birth


def get_pic_url(soup):
    pic_container = soup.find_all('img', class_='thumbCarouselImage--RFBqw')
    pic_url = [img['srcset'].split(', ')[0].split(' ')[0] for img in pic_container if 'srcset' in img.attrs]
    if pic_url == []:
        pic_url = [soup.find_all('source')[0]['srcset'].split(' ')[0]]
    return pic_url


def scrape_pic(scraper_num, start_num, end_num):
    dict = {}
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    # bucket path
    fs = gcsfs.GCSFileSystem(project='ubc-mds-pool')

    for n, url in enumerate(complete_url.iloc[start_num:end_num, 0]):
        try:
            target_url = 'https://www.morphmarket.com' + url
            print(target_url)
            driver.get(target_url)
            time.sleep(3 + np.random.random()) # Loading page might take time
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Extract data
            genes = get_gene(soup=soup)
            origin = get_origin(soup=soup)
            sex = get_sex(soup=soup)
            price = get_price(soup=soup)
            birth = get_birth(soup=soup)
            pic_url = get_pic_url(soup=soup)

            # Download pictures
            for i, picture in enumerate(pic_url):
                pic_code = f'{n + start_num}-{i}'
                gcs_path = f'gs://sam_datastorage/scraper-morphmarket/data/img/{pic_code}.png'
                response = requests.get(picture)
                with fs.open(gcs_path, 'wb') as f:
                    f.write(response.content)
                
                dict[pic_code] = [genes, sex, origin, price, birth, target_url]
                print(f'{pic_code} completed')

            # Save result to csv every 5 pages
            if n % 5 == 0:
                labels_df = pd.DataFrame(dict, index=['genes', 'sex', 'origin', 'price', 'birth', 'url']).T.reset_index()
                file_path = f"gs://sam_datastorage/scraper-morphmarket/data/labels/labels-part{scraper_num}.csv"
                with fs.open(file_path, 'w') as f:
                    labels_df.to_csv(f)
                print(f'labels until {n + start_num} are stored')

        except:
            continue
    driver.quit()


if __name__ == '__main__':
    args = sys.argv[1:]
    
    if len(args) == 3:
        scrape_pic(int(args[0]), int(args[1]), int(args[2]))
    else:
        print("Usage: python scraper-pic.py <scraper_num> <start_num> <end_num>")
        print("Example: python scraper-pic.py 1 0 1000")
