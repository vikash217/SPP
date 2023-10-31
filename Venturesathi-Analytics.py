import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64 
st.set_option('deprecation.showPyplotGlobalUse', False)

BASE_URL = "https://venturesathi-prediction.onrender.com"

logo_image = "Logo.png"

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo_image, width=100)
with col2:
    st.title("Stock Price Prediction")

stock_name = st.selectbox("Select a stock", ("Reliance", "Zomato", "Hdfc", "Tcs"))

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
                table_data = [["Date"] + column_names]  # Include "Date" as the first element in the first row
                for date, prices in zip(future_dates, predict_prices):
                    row = [date] + prices
                    table_data.append(row)

                df = pd.DataFrame(table_data[1:], columns=table_data[0]).rename_axis(None, axis=1)

                st.table(df)
                
                csv_data = df.to_csv(index=True)
                b64 = base64.b64encode(csv_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predicted_prices.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

                plt.figure(figsize=(10, 6))
                for i, column_name in enumerate(column_names):
                    plt.plot(future_dates, [price[i] for price in predict_prices], label=column_name)
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.title(f"Stock Price Trend for {stock_name}")
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot()

                plt.figure(figsize=(10, 6))
                for i, column_name in enumerate(column_names):
                    plt.scatter(future_dates, [price[i] for price in predict_prices], label=column_name)
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.title(f"Stock Price Scatter Plot for {stock_name}")
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot()

            else:
                st.error("Error: Failed to retrieve predicted prices.")
        else:
            st.error("Error: Failed to connect to the server.")
else:
    st.error("Error: The start date must be before the end date.")
