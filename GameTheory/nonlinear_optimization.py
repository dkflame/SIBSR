import numpy as np
import pandas as pd
import os
import torch
from torch.optim import Adam
from torch import optim
import matplotlib.pyplot as plt 
import sympy

# def nonlinearfun(x, a, b, c):
# 	y = a / (b - c*x)
# 	return y

def nonlinearfun(x, a, b, c, d):
	y = (a - b*x) / (c - d*x)
	return y

# def nonlinearfun(x, a, b, c):
# 	y = (a - b*x) / (c - x)
# 	return y

# def nonlinearfun(x, a, b):
# 	y = (a - b*x)  / (9.3 - x)
# 	return y

def get_skeleton(model_skeleton_file):
	data = pd.read_csv(model_skeleton_file)
	skeleton = data['skeleton'][0]
	target = data.at[0, 'target']
	variables = data.at[0, 'variables']
	# skeleton_repr = repr(skeleton)
	w_nums = 0
	for i in skeleton:
		if i == 'w':
			w_nums += 1
	# print(skeleton)
	# print(w_nums)
	return skeleton, w_nums, target, variables

def build_nlo(skeleton, xdata, variables, params):
	# w1 = params[0]
	# w2 = params[1]
	# w3 = params[2]
	# print(xdata[0])
	variables2list = eval(variables)
	for i in range(len(variables2list)):
		exec(variables2list[i]+'=%s'%'xdata[i]')
	for i in range(len(params)):
		weight=f'w{i+1}'
		exec(weight+'=%s'%'params[i]')
		# print(weight)
	y = eval(skeleton)
	return y

def run_regression(training_iters, input_data_file, model_skeleton_file, final_expression_file):
	device = "cpu"
	# w1 = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	# w2 = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	# w3 = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	skeleton, w_nums, target, variables = get_skeleton(model_skeleton_file)
	final_expression = skeleton
	params = []
	for i in range(w_nums):
		params.append(torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device)))
	# print(params)
	# params = [w1, w2, w3]
	# params = [a, b, c]
	optimizer = optim.Adam(params, lr=0.02)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=10000)
	train_X, train_y, training_data = get_data(input_data_file, target, variables)
	xdata = torch.Tensor(np.array(train_X).T)
	ydata = torch.Tensor(np.array(train_y))
	bigger = 100
	bigger_xdata = bigger*xdata
	bigger_ydata = bigger*ydata
	for i in range(training_iters):
		optimizer.zero_grad()
		predict_y = build_nlo(skeleton, xdata, variables, params)
		# predict_y = nonlinearfun(xdata, *params)
		loss = torch.mean(torch.pow(ydata - predict_y, 2))
		if (loss < 1e-10):
			print("early stop at step = %s loss = %s"%(i, loss.detach().numpy()))
			break
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		if i%100 == 0:
			print("step = %s loss = %s"%(i, loss.detach().numpy()))

	params_final = []
	for i in range(len(params)):
		# print(params[i])
		param_i_final = params[i].detach().numpy()[0]
		print(param_i_final)
		params_final.append(param_i_final)
	print("params: ")
	print (params_final)
	for i in range(w_nums):
		weight = 'w' + str(i+1)
		if weight in final_expression:
			final_expression = final_expression.replace(weight, str(params_final[i]))
	final_expression = final_expression.replace('+ -', '-')
	print(final_expression)
	predict_y = build_nlo(skeleton, xdata, variables, params)
	loss = torch.mean(torch.pow(ydata - predict_y, 2))
	mse = loss.detach().numpy()
	print("final loss = %s"%(mse))
	predict_y = build_nlo(skeleton, xdata, variables, params)
	# training_data['pred_x1'] = predict_y.detach().numpy()
	# savedata_file = os.path.join("data/", "game3x3_1vals_b11_c_b12_1_pred.csv")
	# training_data.to_csv(savedata_file, index=False)
	expression_df = [target, variables, str(final_expression), str(mse)]
	df = pd.DataFrame([expression_df], columns=['target', 'variables', 'final_expression', 'mse'])
	df.to_csv(final_expression_file, index=False)

	fig, ax = plt.subplots()
	ax.plot(np.array(train_X).T[0], np.array(train_y), label='truth')
	ax.plot(np.array(train_X).T[0], predict_y.detach().numpy(), label='predict')
	ax.legend()
	plt.show()

def manual_run_regression(training_iters, input_data_file):
	device = "cpu"
	a = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	b = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	c = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	d = torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device))
	# a = torch.nn.Parameter(torch.tensor(5.3))
	# b = torch.nn.Parameter(torch.tensor(0.8))
	# c = torch.nn.Parameter(torch.tensor(7.5))
	params = [a, b, c, d]
	# params = [a, b, c]
	optimizer = optim.Adam(params, lr=0.02)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=10000)
	train_X, train_y, training_data = get_data(input_data_file)
	xdata = torch.Tensor(np.array(train_X))
	ydata = torch.Tensor(np.array(train_y))
	bigger = 100
	bigger_xdata = bigger*xdata
	bigger_ydata = bigger*ydata
	for i in range(training_iters):
		optimizer.zero_grad()
		predict_y = nonlinearfun(xdata, *params)
		loss = torch.mean(torch.pow(ydata - predict_y, 2))
		if (loss < 1e-10):
			print("early stop at step = %s loss = %s"%(i, loss.detach().numpy()))
			break
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		if i%100 == 0:
			print("step = %s loss = %s"%(i, loss.detach().numpy()))

	params_final = []
	for i in range(len(params)):
		# print(params[i])
		param_i_final = params[i].detach().numpy()[0]
		print(param_i_final)
		params_final.append(param_i_final)
	print("params: ")
	print (params_final)
	predict_y = nonlinearfun(xdata, *params)
	loss = torch.mean(torch.pow(ydata - predict_y, 2))
	print("final loss = %s"%(loss.detach().numpy()))
	predict_y = nonlinearfun(xdata, *params)
	training_data['pred_x2'] = predict_y.detach().numpy()
	savedata_file = os.path.join("data/", "game3x3_1vals_nopure_b11_d2_nl_pred.csv")
	training_data.to_csv(savedata_file, index=False)

	fig, ax = plt.subplots()
	ax.plot(np.array(train_X), np.array(train_y), label='truth')
	ax.plot(np.array(train_X), predict_y.detach().numpy(), label='predict')
	ax.legend()
	plt.show()


# Another method
def compute_loss(a, b, c, data):
	total_loss = 0
	# M = len(data)
	# for i in range(M):
	# 	x = data.loc[i]['b11']
	# 	y = data.loc[i]['x1']
	# 	total_loss += (a / (b - c*x) - y)**2
	# loss = total_loss/M
	# print(loss)
	x_df = data['b11']
	y_df = data['x1']
	loss = ((a/(b-c*x_df)-y_df)**2).mean()
	print(loss)
	return loss

def grad_desc(data, init_a, init_b, init_c, alpha, training_iters):
	a = init_a
	b = init_b
	c = init_c
	cost_list = []
	for i in range(training_iters):
		cost_list.append(compute_loss(a, b, c, data))
		a, b, c = step_grad_desc(a, b, c, alpha, data)
	return [a, b, c, cost_list]

def step_grad_desc(current_a, current_b, current_c, alpha, data):
	# sum_grad_a = 0
	# sum_grad_b = 0
	# sum_grad_c = 0
	# M = len(data)
	# for i in range(M):
	# 	x = data.loc[i]['b11']
	# 	y = data.loc[i]['x1']
	# 	sum_grad_a += 2*(current_a/(current_b - current_c*x) - y)
	# 	sum_grad_b += 2*(current_a/(current_b - current_c*x) - y)*(-1)/(current_b-current_c*x)**2
	# 	sum_grad_c += 2*(current_a/(current_b - current_c*x) - y)*(-1)/(current_b-current_c*x)**2*(-1)*x
	# grad_a = sum_grad_a/M
	# grad_b = sum_grad_b/M
	# grad_c = sum_grad_c/M

	x_df = data['b11']
	y_df = data['x1']
	grad_a = (2*(current_a/(current_b - current_c*x_df) - y_df)*(1/(current_b-current_c*x_df))).mean()
	grad_b = (2*(current_a/(current_b - current_c*x_df) - y_df)*(-current_a)/(current_b-current_c*x_df)**2).mean()
	grad_c = (2*(current_a/(current_b - current_c*x_df) - y_df)*(-current_a)/(current_b-current_c*x_df)**2*(-x_df)).mean()
	updated_a = current_a - alpha*grad_a
	updated_b = current_b - alpha*grad_b
	updated_c = current_c - alpha*grad_c
	return updated_a, updated_b, updated_c

def run_regression2(training_iters, input_data_file):
	train_X, train_y, training_data = get_data(input_data_file)
	alpha = 0.2
	init_a = 6.7
	init_b = 28.0
	init_c = 3.0
	a,b,c,cost_list = grad_desc(training_data, init_a, init_b, init_c, alpha, training_iters)
	print("params a, b, c: ")
	print("a = %f b = %f c = %f"%(a, b, c))


# Load data
def load_data(input_file):
	data = pd.read_csv(input_file)
	# data = data.fillna(0)
	print(data.head())
	return data

def getMixStrategyData(data):
	filtered = data[(~data['x1'].isin([0.0, 1.0])) & (~data['x2'].isin([0.0, 1.0])) & (~data['x3'].isin([0.0, 1.0])) & (~data['y1'].isin([0.0, 1.0])) & (~data['y2'].isin([0.0, 1.0])) & (~data['y3'].isin([0.0, 1.0]))]
	return filtered

def get_data(input_data_file, target, variables):
	variables2list = eval(variables)
	input_file = input_data_file
	data = load_data(input_file)
	rows, cols = data.shape
	train_data_rows = int(rows * 0.8)
	test_data_rows = int(rows * 0.2)

	training_data = data.head(train_data_rows)
	testing_data = data.tail(test_data_rows)
	# training_data1 = training_data[(training_data['b11']>9.4) & (training_data['b12']>9.4)]
	training_data1 = getMixStrategyData(training_data)
	training_data1.sort_values(by=variables2list, inplace=True, ascending=True)
	train_y = training_data1[target] 
	# train_y = 1/training_data1[target]  # This is only for reverse function.
	# train_X = training_data1[['b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33']]
	# train_X = training_data1['b11']
	train_X = training_data1[variables2list]
	return train_X, train_y, training_data1

def manual_get_data(input_data_file):
	input_file = input_data_file
	data = load_data(input_file)
	rows, cols = data.shape
	train_data_rows = int(rows * 0.8)
	test_data_rows = int(rows * 0.2)

	training_data = data.head(train_data_rows)
	testing_data = data.tail(test_data_rows)
	# training_data1 = training_data[(training_data['b11']>9.4) & (training_data['b12']>9.4)]
	training_data1 = getMixStrategyData(training_data)
	training_data1.sort_values(by="b11", inplace=True, ascending=True)
	train_y = training_data1['x1'] 
	# train_X = training_data1[['b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33']]
	train_X = training_data1['b11']
	return train_X, train_y, training_data1


if __name__ == "__main__":
	training_iters = 500000
	# input_data_file = 'data/game3x3_9vals.csv'
	# model_skeleton_file = os.path.join("model_result/", "game3x3_9vals_skeleton.csv")
	# final_expression_file = os.path.join("model_result/", "game3x3_9vals_final_expression.csv")

	input_data_file = 'data/game3x3_3vals_b21-23_c_b31_0.csv'
	model_skeleton_file = os.path.join("model_result/", "game3x3_3vals_b21-23_skeleton.csv")
	final_expression_file = os.path.join("model_result/", "game3x3_3vals_b21-23_final_expression.csv")
	# # manual_run_regression(training_iters, input_data_file)
	run_regression(training_iters, input_data_file, model_skeleton_file, final_expression_file)



