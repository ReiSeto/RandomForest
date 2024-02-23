import PySimpleGUI as sg
from joblib import load
from datetime import datetime

# Define the layout of the GUI
model = load('random_forest_model.joblib')
layout = [
    [sg.Text('Enter the values to predict the stock price:')],
    [sg.Text('Date (%dd/%mm/%yyyy):', size=(20,1)), sg.InputText(key='date_input', size=(20,1))],
    [sg.Text('Open:', size=(20,1)), sg.InputText(size=(20,1))],
    [sg.Text('High:', size=(20,1)), sg.InputText(size=(20,1))],
    [sg.Text('Low:', size=(20,1)), sg.InputText(size=(20,1))],
    [sg.Text('Adj Close:', size=(20,1)), sg.InputText(size=(20,1))],
    [sg.Text('Volume:', size=(20,1)), sg.InputText(size=(20,1))],
    [sg.Button('Predict',button_color=('white','#0078D7')) ,
     sg.Button('Cancel', button_color=('white','#0078D7'))]
]

# Create the GUI window
sg.theme('DarkBlue14')
window = sg.Window('Random Forest Regression', layout)
button_color = ('white', '#0078D7')  # Thiết lập màu cho nút

# Event loop
while True:
    event, values = window.Read()
    if event in (None, 'Cancel'):
        break
    elif event == 'Predict':
        # Convert the input values to the correct data type
        date = datetime.strptime(values['date_input'], '%d/%m/%Y').timestamp()
        open_price = float(values[0])
        high_price = float(values[1])
        low_price = float(values[2])
        adj_close_price = float(values[3])
        volume = float(values[4])
        
        # Make the prediction using the trained model
        input_data = [[date, open_price, high_price, low_price, adj_close_price, volume]]
        predicted_price = model.predict(input_data)
        
        # Display the predicted price in a pop-up window
        sg.Popup('The predicted stock price is: {}'.format(predicted_price[0])) 
        
window.Close()