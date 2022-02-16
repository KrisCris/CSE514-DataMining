import pandas as pd
import numpy as np
from math import inf
import matplotlib.pyplot as plt


def matrix_loss(x_ones, y, m_b, size):
    return ((y - x_ones @ m_b) ** 2).sum() / size


def update_alpha(alpha, los, new_los):
    if new_los < los:
        alpha *= 1.01
    else:
        alpha *= 0.5
    return alpha


def prepare_data_univariate(df, idx, start=0, end=900):
    # n * 1 y vector
    prep_y = df[start:end, -1].reshape(-1, 1)
    # n * m matrix
    prep_x = df[start:end, idx].reshape(-1, 1)
    # n * m+1 matrix, last col are ones
    prep_x_ones = np.concatenate([prep_x, np.ones((prep_x.shape[0], 1))], axis=1)
    return [prep_x_ones, prep_y]


def prepare_data_multivariate(df, start=0, end=900):
    # n * 1 y vector
    prep_y = df[start:end, -1].reshape(-1, 1)
    # n * m matrix
    prep_x = df[start:end, :-1]
    # n * m+1 matrix, last col are ones
    prep_x_ones = np.concatenate([prep_x, np.ones((prep_x.shape[0], 1))], axis=1)
    return [prep_x_ones, prep_y]


def prepare_data_polynomial(df, start=0, end=900):
    # n * 1 y vector
    prep_y = df[start:end, -1].reshape(-1, 1)
    # n * m matrix
    prep_x = df[start:end, :-1]
    # concatenate polynomial values to X
    prep_x_45 = prep_x
    for i in range(prep_x.shape[1]):
        for j in range(i, prep_x.shape[1]):
            tmp = (prep_x[:, i] * prep_x[:, j]).reshape(-1, 1)
            prep_x_45 = np.concatenate([prep_x_45, tmp], axis=1)
    prep_x_45 = np.concatenate([prep_x_45, np.ones((prep_x.shape[0], 1))], axis=1)
    return [prep_x_45, prep_y]


def prepare_m_b(nums):
    # m+1 * 1 vector, nums-1 are m, last one is b
    return np.zeros(nums).reshape(-1, 1)


def matrix_derivatives(y, x, w, size):
    # Credit: http://vxy10.github.io/2016/06/25/lin-reg-matrix/
    return (w.T @ x.T @ x - y.T @ x) / size


def standardize(x, scales=None):
    if scales:
        for i in range(len(scales)):
            x[:, i] = (x[:, i] - scales[i][0]) / scales[i][1]
        return x
    else:
        arr = []
        for i in range(x.shape[1] - 1):
            max_val = x[:, i].max()
            min_val = x[:, i].min()
            gap = max_val - min_val
            if gap == 0:
                continue
            x[:, i] = (x[:, i] - min_val) / gap
            arr.append((min_val, gap))
        return x, arr


def regression(x_ones, y, alpha, max_steps, stop_val, debug=False):
    # train phrase
    # x_ones, y = prepared_data
    m_b = prepare_m_b(x_ones.shape[1])

    # init vars
    norm_derivatives = inf
    loss = inf
    steps = 0

    while norm_derivatives > stop_val and steps < max_steps:
        # derivatives of m and b
        derivatives = matrix_derivatives(y=y, x=x_ones, w=m_b, size=len(y))

        # update m and b
        m_b -= (derivatives.T * alpha)

        # l2 norm of vector of partial derivatives of m and b
        norm_derivatives = np.linalg.norm(derivatives)
        steps += 1

        # update learning rate based on loss
        new_loss = matrix_loss(x_ones=x_ones, y=y, m_b=m_b, size=len(y))
        alpha = update_alpha(alpha=alpha, los=loss, new_los=new_loss)
        loss = new_loss
        if debug:
            print(loss)

    return m_b, loss, steps


def r_square(mse, y):
    return 1 - mse / np.var(y)


def plot(x, y, pred_y, name):
    plt.title(f"Feature {name}")
    plt.xlabel("Predictor")
    plt.ylabel("Response")
    plt.scatter(x, y)
    plt.plot(x, pred_y, color="red")
    plt.savefig(f'{name}.png')
    plt.clf()


def train(data_prep, std=False,
          derivative_stop=0.00001, steps_stop=500000, learning_rate=0.000001, save_plt=False, name=None):
    scaler = None
    x, y = data_prep
    x_bk = x
    if std:
        x, scaler = standardize(x)
    params, loss, steps = regression(x_ones=x, y=y, stop_val=derivative_stop, max_steps=steps_stop, alpha=learning_rate)
    print(f"#### STD: {std} ####")
    print("[TRAINING]")
    print(f"Params: {params.reshape(1, -1)[0]}")
    print(f"Loss: {loss}")
    print(f"Steps: {steps}")
    print(f"R Square on Training Set: {r_square(mse=loss, y=y)}\n")
    if save_plt:
        plot(x=x_bk[:, 0].reshape(-1, 1), y=y, pred_y=x @ params, name=name if not std else f'{name}-std')

    return (params, scaler) if std else params


def test(data_prep, params, std=False, scaler=None):
    x, y = data_prep
    loss = matrix_loss(x_ones=x if not std else standardize(x=x, scales=scaler), y=y, m_b=params, size=len(y))
    print("[TESTING]")
    print(f"LOSS: {loss}")
    print(f"R Square: {r_square(mse=loss, y=y)}\n")


if __name__ == '__main__':
    data = pd.read_excel(io='./Concrete_Data.xls', sheet_name='Sheet1').to_numpy()
    train_len = 900
    test_len = 130

    print("######## uni-variate linear regression ########")
    for col in range(len(data[0]) - 1):
        print(f"######## Col: {col} ########")
        # STD
        param, scale = train(data_prep=prepare_data_univariate(df=data, idx=col),
                             std=True, derivative_stop=0.00000001, save_plt=True, name=col)
        test(data_prep=prepare_data_univariate(df=data, idx=col, start=train_len, end=train_len + test_len),
             params=param, std=True, scaler=scale)
        # NO STD
        param = train(data_prep=prepare_data_univariate(df=data, idx=col),
                      derivative_stop=0.00000001, save_plt=True, name=col)
        test(data_prep=prepare_data_univariate(df=data, idx=col, start=train_len, end=train_len + test_len),
             params=param)

    print("######## multi-variate linear regression ########")
    # STD
    param, scale = train(data_prep=prepare_data_multivariate(df=data), std=True)
    test(data_prep=prepare_data_multivariate(df=data, start=train_len, end=train_len + test_len),
         params=param, std=True, scaler=scale)
    # NO STD
    param = train(data_prep=prepare_data_multivariate(df=data))
    test(data_prep=prepare_data_multivariate(df=data, start=train_len, end=train_len + test_len),
         params=param)

    print("######## multi-variate polynomial regression ########")
    # STD
    param, scale = train(data_prep=prepare_data_polynomial(df=data), std=True, derivative_stop=0.000001)
    test(data_prep=prepare_data_polynomial(df=data, start=train_len, end=train_len + test_len),
         params=param, std=True, scaler=scale)
    # NO STD
    param = train(data_prep=prepare_data_polynomial(df=data),
                  derivative_stop=0.000001, learning_rate=0.0000000001)
    test(data_prep=prepare_data_polynomial(df=data, start=train_len, end=train_len + test_len),
         params=param)
