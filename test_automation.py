import webbrowser
import pyautogui
import time
from get_data_from_url import *

def wait_for_page_to_load():
    while True:
        logo = pyautogui.locateOnScreen('images_to_check/loaded_page.png')
        if logo is not None:
            print('web browser ready')
            break

def open_url(url):
    webbrowser.open(url)

def click_on_image(image):
    pyautogui.click('images_to_check/' + image)

url = 'https://www.google.com/'
open_url(url)
wait_for_page_to_load()
print('successfully loaded page')
click_on_image('search_button.png')
print('successfully Clicked')
time.sleep(30)
get_train_dataset_from_url('https://pokemondb.net/pokedex/all')
