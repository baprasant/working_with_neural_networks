import requests
import pandas as pd
import numpy as np
import os

def get_train_dataset_from_url(url,action):
	actions_dict = {'HAND_WAVE' : '1','LEG_WAVE'  : '2'}
	print('\nGetting Training Data for:' + action)
	html = requests.get(url).content
	df_list = pd.read_html(html)
	df = df_list[-1]
	lol = df.values.tolist()
	column1 = list()
	column2 = list()
	column3 = list()
	column4 = list()
	column5 = list()
	column6 = list()
	column7 = list()
	column8 = list()
	column9 = list()
	##Column 1
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column1.append(lol[row_number][0])
	##Column 2
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column2.append(lol[row_number][1])
	##Column 3
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column3.append(lol[row_number][2])
	##Column 4
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column4.append(lol[row_number][3])
	##Column 5
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column5.append(lol[row_number][4])
	##Column 6
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column6.append(lol[row_number][5])
	##Column 7
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column7.append(lol[row_number][6])
	##Column 8
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column8.append(lol[row_number][7])
	##Column 9
	number_of_rows, number_of_columns = np.shape(lol)
	for row_number in range(number_of_rows):
	    column9.append(lol[row_number][8])
	file = open("url_data/train/column1.txt", "a")
	for i in range(len(column1)):
		file.write(str(column1[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column2.txt", "a")
	for i in range(len(column2)):
		file.write(str(column2[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column3.txt", "a")
	for i in range(len(column3)):
		file.write(str(column3[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column4.txt", "a")
	for i in range(len(column4)):
		file.write(str(column4[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column5.txt", "a")
	for i in range(len(column5)):
		file.write(str(column5[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column6.txt", "a")
	for i in range(len(column6)):
		file.write(str(column6[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column7.txt", "a")
	for i in range(len(column7)):
		file.write(str(column7[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column8.txt", "a")
	for i in range(len(column8)):
		file.write(str(column8[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/column9.txt", "a")
	for i in range(len(column9)):
		file.write(str(column9[i]) + ' ')
	file.write('\n')
	file.close()
	file = open("url_data/train/output.txt", "a")
	file.write(actions_dict[action]+'\n')
	file.close()
