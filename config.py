# Hyperparameters and values for plot tuning
config = {
    "alpha_vantage": {
        "key": "demo",
        "symbol": "",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
        "filename": 'data.json'
    }, 
    "plots": {
        "xticks_interval": 90, # showing date every 90 days
        "color_actual": "#4287f5",        # Blue
        "color_train": "#6a8a82",         # Greenish Grey
        "color_val": "#ae82bd",           # Purple
        "color_pred_train": "#6a8a82",    # Greenish Grey, matching color_train for consistency
        "color_pred_val": "#ae82bd",      # Purple
        "color_pred_test": "#e74c3c",     # Red

    },
    "model": {
        "input_size": 1, # as we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}
