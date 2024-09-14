import joblib

def model_save(reg):
    joblib.dump(reg , 'regression_model.joblib')
    return 'regression_model.joblib'

def model_load(new_reg):
    return joblib.load(new_reg)

