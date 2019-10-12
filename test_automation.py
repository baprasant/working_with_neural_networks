import os
import time
import tqdm
from get_data_from_url import *
import urllib

def request_url_silently(url):
    urllib.request.urlopen(url)

#################################################
repetitions = 0
time_steps = 100

reset_url = 'http://172.20.10.10:8266/RESET'
data_url = 'http://172.20.10.10:8266/HOME'
action = 'HAND_WAVE'
no_of_repetitions = 48
for each_time in tqdm.trange(no_of_repetitions,desc = 'Data Collection Progress', leave = False):
    print('Requesting Reset')
    request_url_silently(reset_url)
    print('Reset Done...')
    for collecting_data in tqdm.trange(20, desc = 'Collecting Data', leave = True):
        time.sleep(1)
    get_train_dataset_from_url(data_url, action)
    repetitions +=1
print('Details of Collected Data:')
print('time_steps:' + str(time_steps))
print('total repetitions:' + str(repetitions))
print('total time steps:' + str(repetitions*time_steps))
#################################################
