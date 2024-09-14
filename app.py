from data_cleaning import clean
from model_building import model
from model_io import model_save, model_load
import joblib
import numpy as np
import gradio as gr


file_path = 'Student_Performance.csv'

# Clean the data and train the model
cleaned_data = clean(file_path)
trained_model = model(cleaned_data)
saved_model = model_save(trained_model)
load_model = model_load(saved_model)

# Load the scaler
scaler = joblib.load('scaler.joblib')

def student_performance(hours_st, prev_sco, Extracurricular, sleep, sqpp):
    try:
        # Convert input values to integers
        hours_st = int(hours_st)
        prev_sco = int(prev_sco)
        Extracurricular = int(Extracurricular)  # Assuming this is already transformed to int
        sleep = int(sleep)
        sqpp = int(sqpp)


        input_values_to_scale = [hours_st, prev_sco, sleep, sqpp]
        input_scaled = scaler.transform(np.array(input_values_to_scale).reshape(1, -1))

        input_scaled= input_scaled.flatten().tolist()
        input_scaled.insert(2, Extracurricular)
        prediction = load_model.predict([input_scaled])
        return str(prediction)
    
    except ValueError as e:
        return f"Invalid input. Please enter valid integers. Error: {e}"


iface = gr.Interface(
    fn=student_performance,
    inputs=["number", "number", "number", "number", "number"],
    outputs="text"
)

iface.launch()
