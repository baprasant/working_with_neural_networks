import xlrd
import os

def write_data_to_file(data, path):
    # print(data)
    f = open(path, "a")
    f.write(data)
    f.close()

def convert_to_xl(path):
    import pandas as pd
    import numpy as np
    # Reading the csv file
    df_new = pd.read_csv('Names.csv')
    # saving xlsx file
    GFG = pd.ExcelWriter('Names.xlsx')
    df_new.to_excel(GFG, index = False)
    GFG.save()


current_path = os.getcwd()
action_grading_path = current_path + '/ActionGrading/'
path_for_text_files = current_path + '/ReArrangedData/'
path_for_sensor_1_x = path_for_text_files + 'sensor_1_x.txt'
path_for_sensor_1_y = path_for_text_files + 'sensor_1_y.txt'
path_for_sensor_1_z = path_for_text_files + 'sensor_1_z.txt'
path_for_sensor_2_x = path_for_text_files + 'sensor_2_x.txt'
path_for_sensor_2_y = path_for_text_files + 'sensor_2_y.txt'
path_for_sensor_2_z = path_for_text_files + 'sensor_2_z.txt'
path_for_output = path_for_text_files + 'output.txt'
for grading in range(1,11):
    for experiment_number in range(1,51):
        if grading == 4 and experiment_number == 18:
            print('grading '+str(grading)+' experiment_no '+str(experiment_number)+' skipped')
            continue
        if grading == 4 and experiment_number == 19:
            print('grading '+str(grading)+' experiment_no '+str(experiment_number)+' skipped')
            continue
        if grading == 6 and experiment_number == 16:
            print('grading '+str(grading)+' experiment_no '+str(experiment_number)+' skipped')
            continue
        print('grading '+str(grading)+' experiment_no '+str(experiment_number)+' enetered')
        path = action_grading_path+'Action'+str(grading)+'/AC_'+str(grading)+'_'+str(experiment_number)+'.xlsx'
        print('path:'+path)
        wb = xlrd.open_workbook(path)
        sheet = wb.sheet_by_index(0)
        # print(sheet.cell_value(0,0))
        # row, column
        # print(sheet.nrows)
        # print(sheet.ncols)
        # timestamp sensorid x y z
        for i in range(1,sheet.nrows):
            sensor_id_column = 1
            x_axis_column = 2
            y_axis_column = 3
            z_axis_column = 4
            # print(sheet.cell_value(i,sensor_id_column))
            # print(sheet.cell_value(i,x_axis_column))
            # print(sheet.cell_value(i,y_axis_column))
            # print(sheet.cell_value(i,z_axis_column))
            if str(sheet.cell_value(i,sensor_id_column)) == '1.0':
                # print(sheet.cell_value(i,sensor_id_column))
                write_data_to_file(str(sheet.cell_value(i,x_axis_column)), path_for_sensor_1_x)
                write_data_to_file(' ', path_for_sensor_1_x)
                write_data_to_file(str(sheet.cell_value(i,y_axis_column)), path_for_sensor_1_y)
                write_data_to_file(' ', path_for_sensor_1_y)
                write_data_to_file(str(sheet.cell_value(i,z_axis_column)), path_for_sensor_1_z)
                write_data_to_file(' ', path_for_sensor_1_z)
            if str(sheet.cell_value(i,sensor_id_column)) == '2.0':
                # print(sheet.cell_value(i,sensor_id_column))
                write_data_to_file(str(sheet.cell_value(i,x_axis_column)), path_for_sensor_2_x)
                write_data_to_file(' ', path_for_sensor_2_x)
                write_data_to_file(str(sheet.cell_value(i,y_axis_column)), path_for_sensor_2_y)
                write_data_to_file(' ', path_for_sensor_2_y)
                write_data_to_file(str(sheet.cell_value(i,z_axis_column)), path_for_sensor_2_z)
                write_data_to_file(' ', path_for_sensor_2_z)
        write_data_to_file('\n', path_for_sensor_1_x)
        write_data_to_file('\n', path_for_sensor_1_y)
        write_data_to_file('\n', path_for_sensor_1_z)
        output_data = str(grading)+'\n'
        write_data_to_file(output_data, path_for_output)
