import pandas as pd
import numpy as np
import os
import multiprocessing
# import json
import commentjson as json
import random
import sympy
from copy import deepcopy
from collections import Counter


# Load data
def load_data(input_file):
	data = pd.read_csv(input_file)
	# data = data.fillna(0)
	print(data.head())
	return data

# eq=2*x**2+7*cos(8*y)+2*pi

def nfact2dum(m):
    assert m.is_Mul
    nonnum = sympy.sift(m.args, lambda i:i.is_number, binary=True)[1]
    print(m.is_)
    # return sympy.Mul(*([sympy.Dummy('n')] + nonnum))
    return sympy.Mul(*([1.0] + nonnum))

def float2dum(m):
    assert m.is_Float
    const = sympy.Dummy('w')
    return const

# deq = eq.replace(lambda x:x.is_Mul, lambda x: nfact2dum(x))

# print(deq.subs(list(zip(deq.atoms(Dummy),numbered_symbols('c')))))


def pick_skeleton(model_expression_file):
	df = pd.read_csv(model_expression_file)
	skeleton_list = []
	target = df.at[0, 'target']
	variables = df.at[0, 'variables']
	for i in range(len(df)):
		expression = df.at[i, 'expression']
		expression = expression.replace('[', '')
		expression = expression.replace(']', '')
		expression = expression.strip()
		# print(expression)
		# b11 = sympy.symbols('x1')
		expr = sympy.sympify(expression)
		expr_simple = sympy.simplify(expr)
		expr_simple = sympy.cancel(expr_simple)
		expr_repr = sympy.srepr(expr_simple)
		# expr_dummy = expr_simple.replace(lambda x:x.is_Mul, lambda x: nfact2dum(x))
		expr_dummy = expr_simple.replace(lambda x:x.is_Float, lambda x: float2dum(x))
		expr_dummy_str = str(expr_dummy)
		dummy_repr = sympy.srepr(expr_dummy)
		skeleton_list.append(expr_dummy_str)

	number = Counter(skeleton_list)
	result = number.most_common()
	skeleton = result[0][0]
	print(result[0][1])
	skeleton = skeleton.replace('_', '')
	variables_list = eval(variables)
	for i in range(len(variables_list)):
		skeleton = skeleton.replace('x'+str(i+1), variables_list[i])
	skeleton2list = list(skeleton)
	new_skeleton2list = []
	j=1
	for i in range(len(skeleton2list)):
		if skeleton2list[i] == 'w':
			char2add =  'w'+str(j)
			new_skeleton2list.append(char2add)
			j = j+1
		else:
			char2add = skeleton2list[i]
			new_skeleton2list.append(char2add)		
	skeleton = ''.join(new_skeleton2list)
	return skeleton, target, variables


if __name__ == "__main__":
	model_expression_file = os.path.join("model_result/", "game3x3_3vals_b21-23_c_b31_expression.csv")
	model_skeleton_file = os.path.join("model_result/", "game3x3_3vals_b21-23_skeleton.csv")

	skeleton, target, variables = pick_skeleton(model_expression_file)
	print(skeleton)

	data = [target, variables, str(skeleton)]
	df = pd.DataFrame([data], columns=['target', 'variables', 'skeleton'])

	df.to_csv(model_skeleton_file, index=False)

