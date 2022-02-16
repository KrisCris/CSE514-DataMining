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
    # http://vxy10.github.io/2016/06/25/lin-reg-matrix/
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


def regression(prepared_data, alpha=0.000000000001, max_steps=50000, stop_val=0.001, debug=False):
    # train phrase
    x_45_train, y_train = prepared_data
    m_b = prepare_m_b(x_45_train.shape[1])

    # init vars
    norm_derivatives = inf
    loss = inf
    steps = 0

    while norm_derivatives > stop_val and steps < max_steps:
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
        if debug:
            print(loss)

    return [m_b, loss, steps]


if __name__ == '__main__':
    data = pd.read_excel(io='./Concrete_Data.xls', sheet_name='Sheet1').to_numpy()
    train_len = 900
    test_len = 130

    print("\n######## uni-variate linear regression ########")
    for col in range(len(data[0]) - 1):
        print(f"\n######## using col: [{col}] ########")
        # train phase
        prep_data = prepare_data_univariate(df=data, idx=col)
        prep_data[0], scale = standardize(prep_data[0])
        ret = regression(
            prepared_data=prep_data,
            stop_val=0.00000001,
            max_steps=10000000,
            alpha=0.000001
        )
        print(ret)

        # R Square on Training Set
        y_train_var = np.var(prep_data[1])
        MSE = ret[1]
        rs = 1 - MSE / y_train_var
        print("R Square on Training Set: ", rs)

        # plt
        plt_x = data[0:900, col].reshape(-1, 1)
        plt.scatter(plt_x, prep_data[1])
        plt.plot(plt_x, prep_data[0]@ret[0], color="red")
        plt.savefig(f'{col}_{rs}.png')
        plt.clf()

        # test phase
        x_test, y_test = prepare_data_univariate(df=data, idx=1, start=train_len, end=train_len + test_len)
        test_loss = matrix_loss(x_ones=standardize(x=x_test, scales=scale), y=y_test, m_b=ret[0], size=test_len)
        # test_loss = matrix_loss(x_ones=x_test, y=y_test, m_b=ret[0], size=test_len)
        print(test_loss)
        print("R Square on Testing Set: ", 1 - test_loss / y_train_var)

    ##########

    print("\n######## multi-variate linear regression ########")
    # train phase
    prep_data = prepare_data_multivariate(df=data)
    prep_data[0], scale = standardize(prep_data[0])
    ret = regression(
        prepared_data=prep_data,
        alpha=0.000001
    )
    print(ret)

    # R Square on Training Set
    y_train_var = np.var(prep_data[1])
    MSE = ret[1]
    rs = 1 - MSE / y_train_var
    print("R Square on Training Set: ", rs)

    # test phase
    x_test, y_test = prepare_data_multivariate(df=data, start=train_len, end=train_len + test_len)
    test_loss = matrix_loss(x_ones=standardize(x=x_test, scales=scale), y=y_test, m_b=ret[0], size=test_len)
    print(test_loss)
    print("R Square on Testing Set: ", 1 - test_loss / y_train_var)

    ##########

    print("\n######## multi-variate polynomial regression ########")
    # train phase
    prep_data = prepare_data_polynomial(df=data)
    prep_data[0], scale = standardize(prep_data[0])
    ret = regression(
        prepared_data=prep_data,
        alpha=0.000001
        # debug=True
    )
    print(ret)

    # R Square on Training Set
    y_train_var = np.var(prep_data[1])
    MSE = ret[1]
    rs = 1 - MSE / y_train_var
    print("R Square on Training Set: ", rs)

    # test phase
    x_test, y_test = prepare_data_polynomial(df=data, start=train_len, end=train_len + test_len)
    test_loss = matrix_loss(x_ones=standardize(x=x_test, scales=scale), y=y_test, m_b=ret[0], size=test_len)
    # test_loss = matrix_loss(x_ones=x_test, y=y_test, m_b=ret[0], size=test_len)
    print(test_loss)
    print("R Square on Testing Set: ", 1 - test_loss / y_train_var)
