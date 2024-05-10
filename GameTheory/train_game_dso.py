from dso import DeepSymbolicOptimizer
from dso import DeepSymbolicRegressor
import pandas as pd
import numpy as np
import os
import multiprocessing
# import json
import commentjson as json
import random
from copy import deepcopy

# Load data
def load_data(input_file):
	data = pd.read_csv(input_file)
	# data = data.fillna(0)
	print(data.head())
	return data

def listToString(s):    
    # initialize an empty string
    str1 = ", "  
    # return string  
    return (str1.join(s))

def run_dsr(X, y, const_token_list, k, Ho=None, lowerD_const=None, lowerD_var=None):
	np.random.seed(0)
	## Create the model
	config_json = os.path.join("config/", "game_regression.json")
	new_config_json = os.path.join("config/3x3/", "game_regression_" + str(k) + ".json")
	with open(config_json, encoding='utf-8') as f:
		json_data = json.load(f)
		print(const_token_list)
		json_data["task"]["function_set"] = ["div", "add", "sub", "mul", "const"] + const_token_list
		if Ho is not None:
			json_data["training"]["Ho"] = Ho
			json_data["training"]["lowerD_const"] = lowerD_const
			json_data["training"]["lowerD_var"] = lowerD_var

	with open(new_config_json, 'w') as f:
	    f.write(json.dumps(json_data))

	# model = DeepSymbolicRegressor("./config/game_regression.json")
	model = DeepSymbolicRegressor(new_config_json)
	## Fit the model
	model.fit(X, y)
	## View the best expression
	print(model.program_.sympy_expr)
	return model

def run_model(model_expression_file, train_X, train_y, target, variables, adpt_const, adpt_const_value, const_token_list, k, Ho=None, lowerD_const=None, lowerD_var=None):
	model = run_dsr(train_X.to_numpy(), train_y.to_numpy(), const_token_list, k, Ho, lowerD_const, lowerD_var)
	df = pd.read_csv(model_expression_file)
	# df = pd.DataFrame(columns=['target', 'variables', 'adpt_const', 'adpt_const_value', 'expression', 'traversal', 'NRMSE'])
	data = [str(target), str(variables), str(adpt_const), adpt_const_value, listToString(const_token_list), str(model.program_.sympy_expr), str(model.program_),str(model.program_.NRMSE)]
	df.loc[len(df)] = data
	df.to_csv(model_expression_file, mode="w", header=True, index=False)

	# outputfile = open(model_expression_file,"a")
	# outputfile.write('###### expression: \n')
	# # outputfile.write(str(model.expression_) + '\n')
	# outputfile.write(str(model.program_.sympy_expr) + '\n')
	# outputfile.write('###### pre-order traversal: \n')
	# outputfile.write(str(model.program_) + '\n')
	# outputfile.write('###### training dataset NRMSE: \n')
	# outputfile.write(str(model.program_.NRMSE) + '\n')

def fit_expression():
	model_expression_file = os.path.join("model_result/", "game2x2_1vals_b11_nonfit.txt")
	input_file = 'data/game2x2_1vals_b11_nonfit.csv'
	data = load_data(input_file)
	train_y = data["x1"] 
	train_X = data[['b11']]
	# print(train_X)

	run_model(model_expression_file, train_X, train_y)	

# def piecewise_fit(data, 2, 1):
# 	payoffA_headers = data.columns.str.startswith('a')
# 	case1data = data[payoffA_headers[0] < payoffA_headers[1]]
# 	case2data = data[payoffA_headers[0] > payoffA_headers[1]]
# 	case3data = data[payoffA_headers[0] == payoffA_headers[1]]
# 	return case1data, case2data, case3data

# def piecewise_fit(data, 1, 2):
# 	payoffB_headers = data.columns.str.startswith('b')
# 	case1data = data[payoffB_headers[0] < payoffB_headers[1]]
# 	case2data = data[payoffB_headers[0] > payoffB_headers[1]]
# 	case3data = data[payoffB_headers[0] == payoffB_headers[1]]
# 	return case1data, case2data, case3data

# def piecewise_fit(data, n, m):
# 	payoffA = data[data.columns.str.startswith('a')]
# 	payoffB = data[data.columns.str.startswith('b')]


def getMixStrategyData(data):
	filtered = data[(~data['x1'].isin([0.0, 1.0])) & (~data['x2'].isin([0.0, 1.0])) & (~data['y1'].isin([0.0, 1.0])) & (~data['y2'].isin([0.0, 1.0]))]
	return filtered

def main():
	model_expression_file = os.path.join("model_result/", "game3x3_9vals_b11-33_int_expression.csv")
	df = pd.DataFrame(columns=['target', 'variables', 'adpt_const', 'adpt_const_value', 'const_token_list', 'expression', 'traversal', 'NRMSE'])
	df.to_csv(model_expression_file, index=False)

	variables = ["b11", "b12", "b13", "b21", "b22", "b23", "b31", "b32", "b33"]
	basis_variables = ["ba_1", "ba_2", "ba_3", "ba_4"]
	all_variables = basis_variables + variables
	adpt_const = "b11"
	target = "x2"
	# Ho = ["div","x4","add","sub","sub","x4","sub","sub","add","sub","sub","add","sub","add","add","sub","x1","x2","96.0","x3","x1","x3","90.0","6.0","x1","x2","x2","x1","x2"]

	Ho = ["div", "x1", "add", "add", "x2", "x3", "x4"]
	# Ho = None
	lowerD_const = None
	lowerD_var = None
	# lowerD_const = "58.0"
	# lowerD_var = "x9"
	
	# model_expression_file = os.path.join("model_result/", "game3x3_1vals_nopure_b11_d2.txt")
	# input_file = 'data/game3x3_1vals_nopure_b11_d2.csv'
	for k in range(0,1):
		adpt_const_value = k
		# const_token_list = ["6.0", "62.0", "83.0", "74.0", "55.0", "96.0", "90.0", "67.0", "58.0"]
		const_token_list = []
		# const_token_list.append(str(adpt_const_value))
		input_file = os.path.join("data/", "game3x3_18vals_int.csv")
		# input_file = os.path.join("data/", "game3x3_4vals_b11-21_c_b22_" + str(k) + ".csv")
		data = load_data(input_file)

		data['ba_1'] = 0.274 - 0.00116*data['b32'] - 0.000701*data['b12'] - 0.000595*data['b23'] - 0.000589*data['b22'] - 0.000535*data['b31'] - 0.000457*data['b13'] - 0.000443*data['b33'] - 0.000367*data['b21']
		data['ba_2'] = 1.0 - 0.00330*data['b32'] - 0.00317*data['b23'] - 0.00229*data['b22']
		data['ba_3'] = - 0.00214*data['b31'] - 0.00199*data['b12'] - 0.00184*data['b21']
		data['ba_4'] = - 0.00172*data['b13'] - 0.00125*data['b33'] - 0.000417*data['b11']

		rows, cols = data.shape
		train_data_rows = int(rows * 0.8)
		test_data_rows = int(rows * 0.2)

		training_data = data.head(train_data_rows)
		testing_data = data.tail(test_data_rows)

		# training_data1 = training_data[(training_data['b11']>9.4) & (training_data['b12']>9.4)]
		# training_data1 = getMixStrategyData(training_data)
		train_y = training_data[target] 
		# train_X = training_data1[['b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33']]
		train_X = training_data[all_variables]
		run_model(model_expression_file, train_X, train_y, target, all_variables, adpt_const, adpt_const_value, const_token_list, k, Ho, lowerD_const, lowerD_var)

	# ### Multiprocessing
	# p = multiprocessing.Process(target=run_model,args=(model_expression_file, train_X, train_y,)) 
	# p.start()


if __name__ == "__main__":
	main()
	# fit_expression()


