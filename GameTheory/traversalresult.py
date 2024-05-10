try:
    from dso import cyfunc
except ImportError:
    cyfunc = None

import array
from train_game_dso import load_data
import numpy as np
import os
import json

def python_execute(traversal, X):
    """
    Executes the program according to X using Python.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    Returns
    -------
    y_hats : array-like, shape = [n_samples]
        The result of executing the program on X.
    """

    apply_stack = []

    for node in traversal:
        apply_stack.append([node])

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]

            if token.input_var is not None:
                intermediate_result = X[:, token.input_var]
            else:
                intermediate_result = token(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result

    assert False, "Function should never get here!"
    return None

def cython_execute(traversal, X):
    """
    Execute cython function using given traversal over input X.

    Parameters
    ----------

    traversal : list
        A list of nodes representing the traversal over a Program.
    X : np.array
        The input values to execute the traversal over.

    Returns
    -------

    result : float
        The result of executing the traversal.
    """
    if len(traversal) > 1:
        is_input_var = array.array('i', [t is not None for t in traversal])
        return cyfunc.execute(X, len(traversal), traversal, is_input_var)
    else:
        return python_execute(traversal, X)

# def calyvalue(X_input):
# 	traversal=["sin", "div", "add", "sub", "add", "x2", "add", "sub", "add", "x5", "add", "add", "sub", "x1", "x4", "x1", "x8", "x3", "add", "sub", "x1", "x4", "x2", "x3", "add", "add", "x6", "x2", "x2", "add", "add", "add", "add", "add", "add", "add", "x5", "x3", "x3", "x6", "x2", "add", "x4", "x8", "x2", "x4"]

# 	X_input_ndarray = X_input.values
# 	y_pred = python_execute(traversal, X_input_ndarray)
# 	return y_pred

def calvalue2(data):
    a11 = data['a11']
    a12 = data['a12']
    a21 = data['a21']
    a22 = data['a22']
    b11 = data['b11']
    b12 = data['b12']
    b21 = data['b21']
    b22 = data['b22']
    x1 = b11
    x2 = b12
    x3 = b21
    x4 = b22
    y = 1.00000000000000
    data['x1_pred'] = y
    return data

def picknonfitdata(data):
    fitdata = data[np.sqrt((data['x1_pred']-data['x1'])**2)<=1e-6]
    nonfitdata = data[np.sqrt((data['x1_pred']-data['x1'])**2)>1e-6] 
    return fitdata, nonfitdata

def findminmaxinput(data, x_input):
    min_input = np.min(data[x_input])
    max_input = np.max(data[x_input])
    return min_input, max_input


def cal_NRMSE(input_file):
    data = load_data(input_file)
    x1_pred = data['x1_pred']
    x1 = data['x1']
    var_x1 = np.var(x1)
    NRMSE = np.sqrt(np.mean((x1 - x1_pred)**2)/var_x1)
    return NRMSE


if __name__ == "__main__":

    input_file = 'data/game2x2_2vals_b11_b12.csv'
    data = load_data(input_file)

    expression_json = 'data/game2x2_2vals_b11_b12_expression.json'
    expression = '1.00000000000000'

    data_result = calvalue2(data)
    savedata_file = os.path.join("data/", "game2x2_2vals_b11_b12_result.csv")
    data_result.to_csv(savedata_file, index=False)

    fitdata, nonfitdata = picknonfitdata(data_result)
    min_b11, max_b11 = findminmaxinput(fitdata, 'b11')
    min_b12, max_b12 = findminmaxinput(fitdata, 'b12')
    input_dict = {'b11':[min_b11, max_b11], 'b12':[min_b12, max_b12]}
    expression_dict = {str(expression):input_dict}
    data_json = json.dumps(expression_dict, indent=1)
    tf = open(expression_json,'a', newline='\n')
    tf.write(data_json)
    tf.write('\n')
    # json.dump(expression_dict,tf)
    tf.close()
    savedata_file = os.path.join("data/", "game2x2_2vals_b11_b12_nonfit.csv")
    nonfitdata.to_csv(savedata_file, index=False)

    savedata_file = os.path.join("data/", "game2x2_2vals_b11_b12_fit.csv")
    fitdata.to_csv(savedata_file, index=False)

    # training_data = data.head(40)
    # testing_data = data.tail(10)
    # train_y = training_data["x1"] 
    # train_X = training_data[['a11', 'a12', 'a21', 'a22', 'b11', 'b12', 'b21', 'b22']]
    # test_y = testing_data["x1"] 
    # test_X = testing_data[['a11', 'a12', 'a21', 'a22', 'b11', 'b12', 'b21', 'b22']]

    # testing_data_result = calvalue2(training_data)
    # savedata_file = os.path.join("data/", "game2x2_1vals_b11_train_result.csv")
    # # savedata_file = os.path.join("data/", "game2x2_1vals_pred2.csv")
    # testing_data_result.to_csv(savedata_file, index=False)

    # NRMSE = cal_NRMSE(savedata_file)
    # print("NRMSE: ", NRMSE)
    # print(testing_data_result)













