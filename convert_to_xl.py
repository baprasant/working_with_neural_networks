import os
import pandas as pd
import numpy as np


def convert_csv_to_xl_sheet(path):
    # Reading the csv file
    path_ = path + '.csv'
    df_new = pd.read_csv(path_)
    # saving xlsx file
    path_ = path + '.xlsx'
    GFG = pd.ExcelWriter(path_)
    df_new.to_excel(GFG, index = False)
    GFG.save()


current_path = os.getcwd()
action_grading_path = current_path + '/ActionGrading/'
for grading in range(1,11):
    for experiment_number in range(1,51):
        path = action_grading_path+'Action'+str(grading)+'/AC_'+str(grading)+'_'+str(experiment_number)
        convert_csv_to_xl_sheet(path)
