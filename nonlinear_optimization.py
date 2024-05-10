import numpy as np
import pandas as pd
import os
import torch
from torch.optim import Adam
from torch import optim
import matplotlib.pyplot as plt 
import sympy


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
	# skeleton, w_nums, target, variables = get_skeleton(model_skeleton_file)
	skeleton = 'w1 + w2*MKTMRF + w3*SMB + w4*HML'
	w_nums = 4
	target = 'RET'
	variables = "['MKTMRF', 'SMB', 'HML']"

	final_expression = skeleton
	params = []
	for i in range(w_nums):
		params.append(torch.nn.Parameter(torch.randn(1, dtype=torch.float32, device=device)))
	# params = [w1, w2, w3]
	optimizer = optim.Adam(params, lr=0.02)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=10000)
	train_X, train_y = get_data(input_data_file, target, variables)
	xdata = torch.Tensor(np.array(train_X).T)
	ydata = torch.Tensor(np.array(train_y))
	bigger = 100
	bigger_xdata = bigger*xdata
	bigger_ydata = bigger*ydata
	for i in range(training_iters):
		optimizer.zero_grad()
		predict_y = build_nlo(skeleton, xdata, variables, params)
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

	expression_df = [target, variables, str(final_expression), str(mse)]
	df = pd.DataFrame([expression_df], columns=['target', 'variables', 'final_expression', 'mse'])
	df.to_csv(final_expression_file, index=False)

	# fig, ax = plt.subplots()
	# ax.plot(np.array(train_X).T[0], np.array(train_y), label='truth')
	# ax.plot(np.array(train_X).T[0], predict_y.detach().numpy(), label='predict')
	# ax.legend()
	# plt.show()


# Load data
def load_data(input_file):
	data = pd.read_csv(input_file)
	# data = data.fillna(0)
	print(data.head())
	return data

def get_data(input_data_file, target, variables):
	variables2list = eval(variables)
	input_file = input_data_file
	data = load_data(input_file)
	train_X = data[variables2list]
	train_y = data[target]
	return train_X, train_y


if __name__ == "__main__":
	training_iters = 10000
	conm_num = 1
	input_data_file = 'results/ff_train_result/' + str(conm_num) + '.csv'
	model_skeleton_file = None
	final_expression_file = os.path.join("model_result/", "regressed_expression" + str(conm_num) + ".csv")
	# # manual_run_regression(training_iters, input_data_file)
	run_regression(training_iters, input_data_file, model_skeleton_file, final_expression_file)



