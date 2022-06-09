# https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
# https://practicaldatascience.co.uk/machine-learning/how-to-create-image-datasets-for-machine-learning-models
#https://rubikscode.net/2021/06/21/scraping-images-with-python/
#https://medium.com/analytics-vidhya/a-simple-selenium-image-scrape-from-an-interactive-google-image-search-on-mac-45d403e60d9a

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

import os
import time
import io
import requests
from PIL import Image
import hashlib

#pip install webdriver-manager
import selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager



class GoogleScraper():
    '''Downloades images from google based on the query.
       webdriver - Selenium webdriver
       max_num_of_images - Maximum number of images that we want to download
    '''
    def __init__(self, webdriver:webdriver, max_num_of_images:int):
        self.wd = webdriver
        self.max_num_of_images = max_num_of_images

    def _scroll_to_the_end(self):
        #https://github.com/rmei97/shiba_vs_jindo/blob/master/image_scraper.ipynb
        #wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        #time.sleep(1)
        # Get scroll height
        last_height = wd.execute_script("return document.body.scrollHeight")
        scrolled = 100
        while True:
            if (scrolled >= self.max_num_of_images):
                break
            else:
                scrolled += 100
            # Scroll down to bottom
            wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(2)

            # Calculate new scroll height and compare with last scroll height
            new_height = wd.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                # break #insert press load more
                try:
                    element = wd.find_elements_by_class_name('mye4qd')  # returns list
                    element[0].click()
                except:
                    break
            last_height = new_height

        print("Reached the end of page")

    def _build_query(self, query:str):
        return f"https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={query}&oq={query}&gs_l=img"

    def _get_info(self, query: str):
        image_urls = set()

        wd.get(self._build_query(query))

        self._scroll_to_the_end()

        thumbnails = self.wd.find_elements_by_css_selector("img.Q4LuWd")

        print(f"Found {len(thumbnails)} images...")
        print(f"Getting the links...")

        for img in thumbnails[0:self.max_num_of_images]:
            # We need to click every thumbnail so we can get the full image.
            try:
                img.click()
            except Exception:
                print('ERROR: Cannot click on the image.')
                continue

            images = wd.find_elements_by_css_selector('img.n3VNCb')
            time.sleep(0.3)

            for image in images:
                if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))

        return image_urls

    def download_image(self, folder_path:str, url:str):
        try:
            image_content = requests.get(url).content

        except Exception as e:
            print(f"ERROR: Could not download {url} - {e}")

        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            file = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')

            with open(file, 'wb') as f:
                image.save(f, "JPEG", quality=85)
            print(f"SUCCESS: saved {url} - as {file}")

        except Exception as e:
            print(f"ERROR: Could not save {url} - {e}")

    def scrape_images(self, query:str, folder_path='./images'):
        folder = os.path.join(folder_path,'_'.join(query.lower().split(' ')))

        if not os.path.exists(folder):
            os.makedirs(folder)

        image_info = self._get_info(query)
        print(f"Downloading images...")

        for image in image_info:
            self.download_image(folder, image)


wd = webdriver.Chrome(ChromeDriverManager().install())

wd.get('https://google.com')

gs = GoogleScraper(wd, 10)

gs.scrape_images('test')
#gs.scrape_images('pancake')
#gs.scrape_images('waffle')