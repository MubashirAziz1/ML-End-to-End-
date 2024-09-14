from data_cleaning import clean
from model_building import model
from model_io import model_save , model_load
from fastapi import FastAPI


file_path = 'D:\AI\ML\Cohort\MLOPs\Portfolio\Student_Performance.csv'

cleaned_data = clean(file_path)
trained_model = model(cleaned_data)
saved_model = model_save(trained_model)
load_model = model_load(saved_model)

app = FastAPI()

@app.get('/')
async def home():
    return "Predict the Student Performance with Mubashir Aziz"


