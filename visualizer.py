import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

# plot 1
def plot_daily_close_prices(data_date, data_close_price, num_data_points, display_date_range, config):
    fig = figure(figsize=(25, 5), dpi=80, facecolor=(0.827, 0.839, 0.859))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily timeseries of close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
    plt.grid(which='major', axis='y', linestyle='--')
    plt.show()

# plot 2
def plot_data_split(data_date, config, num_data_points, split_index, scaler, data_y_train, data_y_val):
   # prepare data for plotting

    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)
    
    to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)
    
    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    #------------------------------------
    fig = figure(figsize=(25, 5), dpi=80, facecolor=(0.827, 0.839, 0.859))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily timeseries of close prices for " + config["alpha_vantage"]["symbol"] + " - training and validation data")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

# plot 3
def plot_compare_variation(data_date, data_close_price, num_data_points, config, split_index, scaler, data_y_val, predicted_train, predicted_val):
    # prepare data for plotting
    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    #------------------------------------
    fig = figure(figsize=(25, 5), dpi=80, facecolor=(0.827, 0.839, 0.859))
    plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Comparison of predicted prices against actual prices")
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

# plot 4
def plot_validation_results(data_date, data_y_val, predicted_val, split_index, config, scaler):

    # prepare data for plotting the zoomed in view of the predicted prices (on validation set) vs. actual prices
    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]
    #------------------------------------
    fig = figure(figsize=(25, 5), dpi=80, facecolor=(0.827, 0.839, 0.859))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Zoomable view")
    xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
    xs = np.arange(0,len(xticks))
    plt.xticks(xs, xticks, rotation='vertical')
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

# plot 5
def plot_predicted_prices(plot_range, data_y_val, predicted_val, prediction, scaler, data_date, config):

    # prepare plots 5
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

    to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)

    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    plot_date_test = data_date[-plot_range+1:]
    plot_date_test.append("tomorrow")
    #-----------------------------------
    fig = figure(figsize=(25, 5), dpi=80, facecolor=(0.827, 0.839, 0.859))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Predicted prices of past", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for the next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
    print("\nDate(test):\n")
    print(plot_date_test)
    plt.title("Predicted closing price of the next trading day")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
    
    print("Predicted closing price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))
