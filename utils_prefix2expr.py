from sympy import *
from sympy import parse_expr,sympify

def pre_to_in_order(pre_expr):
	# pattern = {'protectedDiv': '/', 'x1': 'MKTMF', 'x2': 'SMB', 'x3': 'HML', 'add': '+', 'sub': '-', 'mul': '*'}
	# pre_expr = [pattern[x] if x in pattern else x for x in pre_expr]
	stack = []
	for token in reversed(pre_expr):
		if token in ['+', '-', '*', '/']:
			# print(stack)
			op = token
			arg1 = stack.pop()
			arg2 = stack.pop()
			expr = f"({arg1}{op}{arg2})"
			stack.append(expr)
		elif token in ['exp', 'log', 'sin', 'cos']:
			op = token
			arg1 = stack.pop()
			expr = f"{op}({arg1})"
			stack.append(expr)
		else:
			stack.append(str(token))
	return stack[-1]

if __name__ == "__main__":
	# preorder_expr = ['-', '*', '/', 'chmom_B', 'chmom_B', '-', 'mve_S', 'BETA_B', '/', 'mve_B', 'chmom_B']
	preorder_expr = ['*', '+', 'x1', 'x2', 'x3']
	expr = pre_to_in_order(preorder_expr)

	print(expr)

	# x, y, z = symbols('x y z')
	# expr = 2*sin(x)*cos(x)
	# print(simplify(expr))