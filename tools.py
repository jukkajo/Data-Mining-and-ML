import os
import json
import numpy as np
from alpha_vantage.timeseries import TimeSeries 
from torch.utils.data import Dataset

def save_data_as_json(data_date, data_close_price, num_data_points, display_date_range, output_file):
    print("Saving json: ")
    print("data_date: ", data_date)
    print("data_close_price: ", data_close_price.tolist())
    print("num_data_points: ", num_data_points)
    print("display_date_range: ", display_date_range)
    
    data = {
        "data_date": data_date,
        "data_close_price": data_close_price.tolist(),
        "num_data_points": num_data_points,
        "display_date_range": display_date_range
    }
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file)
        
def download_data(config):
    # Checking if stored, if not -> download and save, for limited request rate reasons   
    if os.path.exists(config["data"]["filename"]):
       print("Opening existing file...")
       with open(config["data"]["filename"], 'r') as json_file:
            data = json.load(json_file)
            data_date = data["data_date"]
            data_close_price = np.array(data["data_close_price"])
            num_data_points = data["num_data_points"]
            display_date_range = data["display_date_range"]
            return data_date, data_close_price, num_data_points, display_date_range
    
    else: 
        print("Fetching data...")
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
        data_date = [date for date in data.keys()]
        data_date.reverse()

        data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
        data_close_price.reverse()
        data_close_price = np.array(data_close_price)

        num_data_points = len(data_date)
        display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
        print("Number data points", num_data_points, display_date_range)
        
        save_data_as_json(data_date, data_close_price, num_data_points, display_date_range, config["data"]["filename"])
      
        return data_date, data_close_price, num_data_points, display_date_range

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
        
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
