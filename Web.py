import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64 
st.set_option('deprecation.showPyplotGlobalUse', False)

BASE_URL = "https://spp-oaaw.onrender.com"

st.title("Stock Price Prediction")

stock_name = st.selectbox("Select a stock", ("Reliance", "Zomato", "HDFC", "TCS"))

start_date = st.date_input("Select start date")
end_date = st.date_input("Select end date")

if start_date < end_date:
    if st.button("Predict"):
        payload = {"stock_name": stock_name, "start_date": str(start_date), "end_date": str(end_date)}

        response = requests.post(f"{BASE_URL}/LSTM_Predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            if "future_prediction" in data:
                predict_prices = data["future_prediction"]
                future_dates = data["future_dates"]
                column_names = data["column_names"]

                st.write(f"Predicted prices for {stock_name} between {start_date} and {end_date}:")
                table_data = [["Date"] + column_names] 
                for date, prices in zip(future_dates, predict_prices):
                    row = [date] + prices
                    table_data.append(row)

                df = pd.DataFrame(table_data[1:], columns=table_data[0]).rename_axis(None, axis=1)

                st.table(df)

            else:
                st.error("Error: Failed to retrieve predicted prices.")
        else:
            st.error("Error: Failed to connect to the server.")
else:
    st.error("Error: The start date must be before the end date.")
