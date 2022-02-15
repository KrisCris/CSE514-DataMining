import pandas as pd
import numpy as np
from math import inf


def matrix_loss(x_ones, y, m_b, size):
    return ((y - x_ones @ m_b) ** 2).sum() / size


def update_alpha(alpha, los, new_los):
    if new_los < los:
        alpha *= 1.01
    else:
        alpha *= 0.5
    return alpha


def prepare_data(df, start=0, end=900):
    # n * 1 y vector
    prep_y = df[start:end, -1].reshape(-1, 1)
    # n * m matrix
    prep_x = df[start:end, :-1]
    # n * m+1 matrix, last col are ones
    prep_x_ones = np.concatenate([prep_x, np.ones((prep_x.shape[0], 1))], axis=1)
    return prep_x_ones, prep_y


def prepare_m_b(nums):
    # m+1 * 1 vector, nums-1 are m, last one is b
    return np.zeros(nums).reshape(-1, 1)


def matrix_derivatives(y, x_ones, m_b, size):
    # http://vxy10.github.io/2016/06/25/lin-reg-matrix/
    return (m_b.T @ x_ones.T @ x_ones - y.T @ x_ones) / size


def multivariable_regression(prepared_data):
    # train phrase
    x_ones_train, y_train = prepared_data
    m_b = prepare_m_b(cols)

    # init vars
    norm_derivatives = inf
    loss = inf
    alpha = 0.000001
    steps = 0
    max_steps = 50000

    while norm_derivatives > 0.01 and steps < max_steps:
        new_loss = matrix_loss(x_ones_train, y_train, m_b, train_len)

        derivatives = matrix_derivatives(y=y_train, x_ones=x_ones_train, m_b=m_b, size=train_len)
        m_b -= (derivatives.T * alpha)
        norm_derivatives = np.linalg.norm(derivatives)
        steps += 1

        alpha = update_alpha(alpha=alpha, los=loss, new_los=new_loss)
        loss = new_loss

    return m_b, loss, steps


if __name__ == '__main__':
    data = pd.read_excel('./Concrete_Data.xls').to_numpy()
    train_len = 900
    test_len = 130
    cols = 9

    train_mb, train_los, step = multivariable_regression(prepare_data(df=data))
    print("train m, b", train_mb.reshape(1, -1)[0])
    print("train loss", train_los)
    print("steps", step)

    # test phase
    x_ones_test, y_test = prepare_data(df=data, start=train_len, end=train_len + test_len)
    print("test loss", matrix_loss(x_ones=x_ones_test, y=y_test, m_b=train_mb, size=test_len))
