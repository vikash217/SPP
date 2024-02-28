import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from keras.models import load_model
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

class StockRequest(BaseModel):
    stock_name: str
    start_date: str
    end_date: str

STOCK_FILE_PATHS = {
#    "Reliance": "RELIANCE.csv",
   "Zomato": "ZOMATO.csv",
#    "Hdfc": "HDFC.csv",
#    "Tcs": "TCS.csv",
}

@app.post("/LSTM_Predict")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    start_date = pd.to_datetime(stock_request.start_date)
    end_date = pd.to_datetime(stock_request.end_date)

    try:
        file_path = STOCK_FILE_PATHS[stock_name]
        df = pd.read_csv(file_path)
    except KeyError:
        raise HTTPException(status_code=422, detail="Invalid stock name")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    data = df.filter(["Open", "High", "Low", "Close"])
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    seq_len = 60
    train_size = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - seq_len:]

    x_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        x_train.append(train_data[i - seq_len: i])
        y_train.append(train_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = load_model('Model.h5')

    future_dates = pd.date_range(start=start_date, end=end_date)
    future_dates = future_dates[future_dates.weekday < 5]  

    x_future = scaled_data[-seq_len:]
    predictions = []

    for date in future_dates:
        if date.weekday() >= 5:
            continue

        x_future_reshaped = np.reshape(x_future, (1, x_future.shape[0], x_future.shape[1]))
        prediction = model.predict(x_future_reshaped)
        predictions.append(prediction[0])
        x_future = np.concatenate((x_future[1:], prediction))

    predictions = scaler.inverse_transform(predictions)

    response = {
        "future_prediction": predictions.tolist(),
        "future_dates": future_dates.strftime("%Y-%m-%d").tolist(),
        "column_names": data.columns.tolist()
    }

    return response
