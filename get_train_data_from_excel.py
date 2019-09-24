from pyexcel_xls import *
import json
import collections
import os

working_dir = os.getcwd()
hand_wave_filepath = working_dir + "/dataset/hand_wave.xlsx"
leg_wave_filepath = working_dir + "/dataset/leg_wave.xlsx"
hand_wave_data = get_data(hand_wave_filepath)
hand_wave_json_str = json.dumps(hand_wave_data)
hand_wave_json = json.loads(hand_wave_json_str) # hand_wave_json has entire book in json format
leg_wave_data = get_data(leg_wave_filepath)
leg_wave_json_str = json.dumps(leg_wave_data)
leg_wave_json = json.loads(leg_wave_json_str) # leg_wave_json has entire book in json format
sheet_list = ['HW_S_1','HW_S_2','HW_S_3','HW_S_4','HW_S_5','HW_S_6','HW_S_7','HW_S_8','HW_S_9','HW_F_1','HW_F_2','HW_F_3','HW_F_4','HW_F_5','HW_F_6','HW_F_7','HW_F_8','HW_F_9']
x_axis_acc = list()
y_axis_acc = list()
z_axis_acc = list()
x_axis_gyro = list()
y_axis_gyro = list()
z_axis_gyro = list()
x_axis_mag = list()
y_axis_mag = list()
z_axis_mag = list()
for sheet in sheet_list:
	for i in range(1,1000):
		x_axis_acc.append(hand_wave_json[sheet][i][1]) # 1 denotes x_axis_acc

for sheet in sheet_list:
	for i in range(1,1000):
		y_axis_acc.append(hand_wave_json[sheet][i][2]) # 2 denotes y_axis_acc

for sheet in sheet_list:
	for i in range(1,1000):
		z_axis_acc.append(hand_wave_json[sheet][i][3]) # 3 denotes z_axis_acc

for sheet in sheet_list:
	for i in range(1,1000):
		x_axis_gyro.append(hand_wave_json[sheet][i][4]) # 4 denotes x_axis_gyro

for sheet in sheet_list:
	for i in range(1,1000):
		y_axis_gyro.append(hand_wave_json[sheet][i][5]) # 5 denotes y_axis_gyro

for sheet in sheet_list:
	for i in range(1,1000):
		z_axis_gyro.append(hand_wave_json[sheet][i][6]) # 6 denotes z_axis_gyro

for sheet in sheet_list:
	for i in range(1,1000):
		x_axis_mag.append(hand_wave_json[sheet][i][7]) # 7 denotes x_axis_mag

for sheet in sheet_list:
	for i in range(1,1000):
		y_axis_mag.append(hand_wave_json[sheet][i][8]) # 8 denotes y_axis_mag

for sheet in sheet_list:
	for i in range(1,1000):
		z_axis_mag.append(hand_wave_json[sheet][i][9]) # 9 denotes z_axis_mag
###################################
file = open("movement_dataset/train/hand_wave/x_axis_acc.txt", "w")
k = 0
for i in x_axis_acc:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/hand_wave/y_axis_acc.txt", "w")
k = 0
for i in y_axis_acc:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/hand_wave/z_axis_acc.txt", "w")
k = 0
for i in z_axis_acc:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
###################################
file = open("movement_dataset/train/hand_wave/x_axis_gyro.txt", "w")
k = 0
for i in x_axis_gyro:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/hand_wave/y_axis_gyro.txt", "w")
k = 0
for i in y_axis_gyro:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/hand_wave/z_axis_gyro.txt", "w")
k = 0
for i in z_axis_gyro:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
###################################
file = open("movement_dataset/train/hand_wave/x_axis_mag.txt", "w")
k = 0
for i in x_axis_mag:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/hand_wave/y_axis_mag.txt", "w")
k = 0
for i in y_axis_mag:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/hand_wave/z_axis_mag.txt", "w")
k = 0
for i in z_axis_mag:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
sheet_list = ['PT_CP_1','PT_CP_2','PT_CP_3','PT_CP_4','PT_CP_5','PT_CP_6','PT_CP_7','PT_CP_8','PT_CP_9','PT_TD_1','PT_TD_2','PT_TD_3','PT_TD_4','PT_TD_5','PT_TD_6','PT_TD_7','PT_TD_8','PT_TD_9']
x_axis_acc = list()
y_axis_acc = list()
z_axis_acc = list()
x_axis_gyro = list()
y_axis_gyro = list()
z_axis_gyro = list()
x_axis_mag = list()
y_axis_mag = list()
z_axis_mag = list()
for sheet in sheet_list:
	for i in range(1,1000):
		x_axis_acc.append(leg_wave_json[sheet][i][1]) # 1 denotes x_axis_acc

for sheet in sheet_list:
	for i in range(1,1000):
		y_axis_acc.append(leg_wave_json[sheet][i][2]) # 2 denotes y_axis_acc

for sheet in sheet_list:
	for i in range(1,1000):
		z_axis_acc.append(leg_wave_json[sheet][i][3]) # 3 denotes z_axis_acc

for sheet in sheet_list:
	for i in range(1,1000):
		x_axis_gyro.append(leg_wave_json[sheet][i][4]) # 4 denotes x_axis_gyro

for sheet in sheet_list:
	for i in range(1,1000):
		y_axis_gyro.append(leg_wave_json[sheet][i][5]) # 5 denotes y_axis_gyro

for sheet in sheet_list:
	for i in range(1,1000):
		z_axis_gyro.append(leg_wave_json[sheet][i][6]) # 6 denotes z_axis_gyro

for sheet in sheet_list:
	for i in range(1,1000):
		x_axis_mag.append(leg_wave_json[sheet][i][7]) # 7 denotes x_axis_mag

for sheet in sheet_list:
	for i in range(1,1000):
		y_axis_mag.append(leg_wave_json[sheet][i][8]) # 8 denotes y_axis_mag

for sheet in sheet_list:
	for i in range(1,1000):
		z_axis_mag.append(leg_wave_json[sheet][i][9]) # 9 denotes z_axis_mag
###################################
file = open("movement_dataset/train/leg_wave/x_axis_acc.txt", "w")
k = 0
for i in x_axis_acc:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/leg_wave/y_axis_acc.txt", "w")
k = 0
for i in y_axis_acc:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/leg_wave/z_axis_acc.txt", "w")
k = 0
for i in z_axis_acc:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
###################################
file = open("movement_dataset/train/leg_wave/x_axis_gyro.txt", "w")
k = 0
for i in x_axis_gyro:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/leg_wave/y_axis_gyro.txt", "w")
k = 0
for i in y_axis_gyro:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/leg_wave/z_axis_gyro.txt", "w")
k = 0
for i in z_axis_gyro:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
###################################
file = open("movement_dataset/train/leg_wave/x_axis_mag.txt", "w")
k = 0
for i in x_axis_mag:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/leg_wave/y_axis_mag.txt", "w")
k = 0
for i in y_axis_mag:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
file = open("movement_dataset/train/leg_wave/z_axis_mag.txt", "w")
k = 0
for i in z_axis_mag:
    if k!= 0 and k%100 == 0:
        file.write("\n"+str(i)+" ")
        k+=1
    else:
        file.write(str(i)+" ")
        k+=1
file.close()
###################################
