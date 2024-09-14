import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean(filepath):
    loaded_file = pd.read_csv(filepath)
    
    # No missing values in the dataset

    # Extracurricular Column in the dataset in categorical object, so it needs to be transformed 
    # into integer dtype variable.
    label = LabelEncoder()
    loaded_file['Extracurricular Activities'] = label.fit_transform(loaded_file['Extracurricular Activities'])

    # Values in all the columns except target column and extracurricluar column differ in their ranges, 
    # so it needs to be scaled in between the specified range.
    scaler = StandardScaler()
    loaded_file[['Hours Studied', 'Previous Scores', "Sleep Hours", "Sample Question Papers Practiced"]] = scaler.fit_transform(loaded_file[['Hours Studied', 'Previous Scores', "Sleep Hours", "Sample Question Papers Practiced"]])

    loaded_file.to_csv('clean_data.csv' , index = False)

