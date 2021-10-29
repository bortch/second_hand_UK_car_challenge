# ----------------------------------------------------------------------------
# Created By  : Bortch - JBS
# Created Date: 09/01/2021
# version ='1.0'
# source = https://github.com/bortch/second_hand_UK_car_challenge
# modification for kaggle
# ---------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import mean_squared_error
import bs_lib.bs_terminal as terminal

def get_rmse(y_prediction,y_real, verbose=False):
    y_exp = np.exp(y_real)
    y_prediction_exp = np.exp(y_prediction)
    rmse = np.sqrt(mean_squared_error(y_exp, y_prediction_exp))
    if verbose:
        print(f"RMSE: {rmse}")
    return y_prediction_exp, y_exp, rmse

def evaluate_prediction(model, X, y, sample=None,add_columns=[]):
    if sample:
        X = X.sample(n=sample, random_state=1)
        y_log = y.sample(n=sample, random_state=1)
    y_pred_log = model.predict(X)

    y_pred, y, rmse = get_rmse(y_pred_log,y_log,verbose=True)

    data = []

    for i in range(len(y_pred)):
        row = []
        sample = X[i:i+1]
        pred = y_pred[i:i+1][0]
        real = int(y[i:i+1].values[0])
        error = (real-pred)
        percentage = (error/real)*100
        row.append(f"{pred:.0f}")
        row.append(f"{real:.0f}")
        row.append(f"{error:.0f}")
        row.append(f"{percentage:.0f} %")
        for col in add_columns:
            value = sample[col].array[0]
            row.append(f"{value}")
        data.append(row)

    table_columns = ['Prediction', 'Real Price', 'Error', 'Percentage']+add_columns 
    table = terminal.create_table(title="Prediction results",
                                  columns=table_columns,
                                  data=data)
    terminal.article(title="Model Prediction testing", content=table)
