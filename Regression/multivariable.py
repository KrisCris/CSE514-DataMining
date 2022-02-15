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


def prepare_data_multivariate(df, start=0, end=900):
    # n * 1 y vector
    prep_y = df[start:end, -1].reshape(-1, 1)
    # n * m matrix
    prep_x = df[start:end, :-1]
    # n * m+1 matrix, last col are ones
    prep_x_ones = np.concatenate([prep_x, np.ones((prep_x.shape[0], 1))], axis=1)
    return prep_x_ones, prep_y


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
    return prep_x_45, prep_y


def prepare_m_b(nums):
    # m+1 * 1 vector, nums-1 are m, last one is b
    return np.zeros(nums).reshape(-1, 1)


def matrix_derivatives(y, x, w, size):
    # http://vxy10.github.io/2016/06/25/lin-reg-matrix/
    return (w.T @ x.T @ x - y.T @ x) / size


def standardize(x):
    for i in range(x.shape[1]-1):
        max_val = x[:, i].max()
        min_val = x[:, i].min()
        if max_val == min_val:
            continue
        x[:, i] = (x[:, i] - min_val) / (max_val-min_val)
    return x



def multivariable_regression(prepared_data):
    # train phrase
    x_ones_train, y_train = prepared_data
    m_b = prepare_m_b(x_ones_train.shape[1])

    # init vars
    norm_derivatives = inf
    loss = inf
    alpha = 0.000001
    steps = 0
    max_steps = 50000

    while norm_derivatives > 0.01 and steps < max_steps:
        # derivatives of m and b
        derivatives = matrix_derivatives(y=y_train, x=x_ones_train, w=m_b, size=train_len)

        # update m and b
        m_b -= (derivatives.T * alpha)

        # l2 norm of vector of partial derivatives of m and b
        norm_derivatives = np.linalg.norm(derivatives)
        steps += 1

        # update learning rate based on loss
        new_loss = matrix_loss(x_ones_train, y_train, m_b, train_len)
        alpha = update_alpha(alpha=alpha, los=loss, new_los=new_loss)
        loss = new_loss

    return m_b, loss, steps


def multivariate_polynomial_regression(prepared_data, standard=False, alpha=0.000000000001, max_steps=500000):
    # train phrase
    x_45_train, y_train = prepared_data
    if standard:
        x_45_train = standardize(x_45_train)
    m_b = prepare_m_b(x_45_train.shape[1])

    # init vars
    norm_derivatives = inf
    loss = inf
    steps = 0

    while norm_derivatives > 0.01 and steps < max_steps:
        # derivatives of m and b
        derivatives = matrix_derivatives(y=y_train, x=x_45_train, w=m_b, size=train_len)

        # update m and b
        m_b -= (derivatives.T * alpha)

        # l2 norm of vector of partial derivatives of m and b
        norm_derivatives = np.linalg.norm(derivatives)
        steps += 1

        # update learning rate based on loss
        new_loss = matrix_loss(x_45_train, y_train, m_b, train_len)
        alpha = update_alpha(alpha=alpha, los=loss, new_los=new_loss)
        loss = new_loss
        print(loss)

    return m_b, loss, steps


if __name__ == '__main__':
    data = pd.read_excel('./Concrete_Data.xls').to_numpy()
    train_len = 900
    test_len = 130
    # cols = 9

    # train_mb, train_los, step = multivariable_regression(prepare_data_multivariate(df=data))
    # print("train m, b", train_mb.reshape(1, -1)[0])
    # print("train loss", train_los)
    # print("steps", step)
    #
    # # test phase
    # x_ones_test, y_test = prepare_data_multivariate(df=data, start=train_len, end=train_len + test_len)
    # print("test loss", matrix_loss(x_ones=x_ones_test, y=y_test, m_b=train_mb, size=test_len))

    print(multivariate_polynomial_regression(
        prepared_data=prepare_data_polynomial(df=data),
        standard=True
    ))
