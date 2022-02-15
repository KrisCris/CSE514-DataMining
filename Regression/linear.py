import pandas as pd


def loss(m, b, x, y, n, nn=0):
    summation = 0
    for i in range(nn, n):
        summation += (y[i] - (m * x[i] + b)) ** 2
    return summation / (n - nn)


def derivative_m(m, b, x, y, n):
    summation = 0
    for i in range(n):
        summation += -2 * x[i] * (y[i] - (m * x[i] + b))
    return summation / n


def derivative_b(m, b, x, y, n):
    summation = 0
    for i in range(n):
        summation += -2 * (y[i] - (m * x[i] + b))
    return summation / n


def update_alpha(alpha, old_los, los):
    if los > old_los:
        return alpha * 0.5
    return alpha * 1.01


def linear_regression(m, b, x, y, n, alpha):
    stop_steps = 1000
    steps = 0
    dm = derivative_m(m, b, x, y, n)
    db = derivative_b(m, b, x, y, n)
    los = loss(m, b, x, y, n)
    while (abs(dm) + abs(db)) > 0.001 and steps < stop_steps:
        # new_dm = derivative_m(m, b, x, y, n)
        # new_db = derivative_b(m, b, x, y, n)
        #
        # alpha = update_alpha(alpha, dm, new_dm, db, new_db)

        dm = derivative_m(m, b, x, y, n)
        db = derivative_b(m, b, x, y, n)
        tmp_los = loss(m, b, x, y, n)
        alpha = update_alpha(alpha, los, tmp_los)
        los = tmp_los
        m -= alpha * dm
        b -= alpha * db

        print("######" + str(steps))
        print('m:' + str(m))
        print('b:' + str(b))
        print('dm:'+str(dm))
        print('db:'+str(db))
        print('lost:' + str(loss(m, b, x, y, n)))
        steps += 1

    return m, b

def univariable():
    pass


def multivariable():
    pass

if __name__ == '__main__':
    data = pd.read_excel('./Concrete_Data.xls')
    rows = len(data.index)
    cols = len(data.columns)

    train_len = 900
    test_len = 130

    # for col in range(cols - 1):
    col = 6
    ret = linear_regression(
        m=0, b=0,
        x=data[data.columns[col]], y=data[data.columns[cols - 1]],
        n=train_len, alpha=0.0000001
    )
    print(ret)
    print(loss(
        m=ret[0], b=ret[1],
        x=data[data.columns[col]], y=data[data.columns[cols - 1]],
        n=test_len+train_len, nn=train_len
    ))
    pass

