import numpy as np
import nashpy as nash
import pandas as pd
import os

# pd.set_option('display.max_columns', None)

# pd.set_option('display.max_rows', None)

# pd.set_option('max_colwidth',100)


def build_2x2game(row, col, savedata_file):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a21', 'a22', 'b11', 'b12', 'b21', 'b22', 'x1', 'x2', 'y1', 'y2'])
	for i in range(500):
		# a1 = np.random.randint(0,10)
		# a2 = np.random.randint(0,10)
		# A[0][0] = a1
		# B[0][0] = a2
		A = np.random.uniform(0.0, 10.0, (row, col))
		B = np.random.uniform(0.0, 10.0, (row, col))
		# A[0][0] = 2.4
		# A[0][1] = 8.6
		# A[1][0] = 7.8
		# A[1][1] = 5
		# B[0][0] = 6.6
		# B[0][1] = 1.6
		# B[1][0] = 1
		# B[1][1] = 9.4
		a11 = A[0][0]
		a12 = A[0][1]
		a21 = A[1][0]
		a22 = A[1][1]
		b11 = B[0][0]
		b12 = B[0][1]
		b21 = B[1][0]
		b22 = B[1][1]
		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		eq1 = list(eqs)[0]
		x1 = eq1[0][0]
		x2 = eq1[0][1]
		y1 = eq1[1][0]
		y2 = eq1[1][1]
		df_append = pd.DataFrame([[a11, a12, a21, a22, b11, b12, b21, b22, x1, x2, y1, y2]], columns=['a11', 'a12', 'a21', 'a22', 'b11', 'b12', 'b21', 'b22', 'x1', 'x2', 'y1', 'y2'])
		df = df.append(df_append)
	# df = df[~df['x1'].isin([0.0, 1.0])]
	df = df[(~df['x1'].isin([0.0, 1.0])) & (~df['x2'].isin([0.0, 1.0])) & (~df['y1'].isin([0.0, 1.0])) & (~df['y2'].isin([0.0, 1.0]))]
	df.to_csv(savedata_file, index=False)		


def build_2x2game_edge_case(row, col, savedata_file):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a21', 'a22', 'b11', 'b12', 'b21', 'b22', 'x1', 'x2', 'y1', 'y2'])
	for i in range(1):
		A = np.random.uniform(0.0, 10.0, (row, col))
		B = np.random.uniform(0.0, 10.0, (row, col))
		A[0][0] = 2
		A[0][1] = 5
		A[1][0] = 2
		A[1][1] = 5
		B[0][0] = 3
		B[0][1] = 10
		B[1][0] = 6
		B[1][1] = 8
		a11 = A[0][0]
		a12 = A[0][1]
		a21 = A[1][0]
		a22 = A[1][1]
		b11 = B[0][0]
		b12 = B[0][1]
		b21 = B[1][0]
		b22 = B[1][1]
		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		print(list(eqs))


def build_2x3game(row, col, savedata_file):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23',  'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'x1', 'x2', 'y1', 'y2', 'y3'])
	for i in range(500):
		# a1 = np.random.randint(0,10)
		# a2 = np.random.randint(0,10)
		# A[0][0] = a1
		# B[0][0] = a2
		A = np.random.uniform(0.0, 10.0, (row, col))
		B = np.random.uniform(0.0, 10.0, (row, col))

		# A[0][0] = 2.4
		# A[0][1] = 4.6
		# A[0][2] = 9
		# A[1][0] = 9.8
		# A[1][1] = 4.2
		# A[1][2] = 1.6

		# # B[0][0] = 7
		# B[0][1] = 4.2
		# B[0][2] = 1.4
		# B[1][0] = 3.2
		# B[1][1] = 2.4
		# B[1][2] = 8.6

		a11 = A[0][0]
		a12 = A[0][1]
		a13 = A[0][2]
		a21 = A[1][0]
		a22 = A[1][1]
		a23 = A[1][2]

		b11 = B[0][0]
		b12 = B[0][1]
		b13 = B[0][2]
		b21 = B[1][0]
		b22 = B[1][1]
		b23 = B[1][2]

		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		# eq1 = None
		# for eq in eqs:
		# 	eq1 = eq
		eqs_list = list(eqs)
		# print(eqs_list)
		for i in range(len(eqs_list)):
			eq1 = eqs_list[i]
			x1 = eq1[0][0]
			x2 = eq1[0][1]
			y1 = eq1[1][0]
			y2 = eq1[1][1]
			y3 = eq1[1][2]
			df_append = pd.DataFrame([[a11, a12, a13, a21, a22, a23, b11, b12, b13, b21, b22, b23, x1, x2, y1, y2, y3]], columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'x1', 'x2', 'y1', 'y2', 'y3'])
			df = df.append(df_append)
	# df = df[(~df['x1'].isin([0.0, 1.0]))&(~df['x2'].isin([0.0, 1.0]))&(~df['y1'].isin([0.0, 1.0]))&(~df['y2'].isin([0.0, 1.0]))&(~df['y3'].isin([0.0, 1.0]))]
	# df = df[(df['x1']>1e-10)&(df['x2']>1e-10)&(df['y1']>1e-10)&(df['y2']>1e-10)&(df['y3']>1e-10)]
	df = df[(df['y2']<1e-10)]
	df.to_csv(savedata_file, index=False)


def build_3x3game(row, col, savedata_file, k):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3'])
	for i in range(500):
		# a1 = np.random.randint(0,10)
		# a2 = np.random.randint(0,10)
		# A[0][0] = a1
		# B[0][0] = a2
		A = np.random.uniform(0.0, 10.0, (row, col))
		B = np.random.uniform(0.0, 10.0, (row, col))
		A[0][0] = 5.2
		A[0][1] = 3.3
		A[0][2] = 2.1
		A[1][0] = 3.2
		A[1][1] = 5
		A[1][2] = 1.2
		A[2][0] = 1.5
		A[2][1] = 4.5
		A[2][2] = 5.3
		# B[0][0] = 0.02*k+0.6
		# B[0][1] = 6.2
		# B[0][2] = 8.3
		B[1][0] = 7.4
		B[1][1] = 5.5
		B[1][2] = 9.6
		B[2][0] = 9
		B[2][1] = 6.7
		B[2][2] = 5.8

		a11 = A[0][0]
		a12 = A[0][1]
		a13 = A[0][2]
		a21 = A[1][0]
		a22 = A[1][1]
		a23 = A[1][2]
		a31 = A[2][0]
		a32 = A[2][1]
		a33 = A[2][2]

		b11 = B[0][0]
		b12 = B[0][1]
		b13 = B[0][2]
		b21 = B[1][0]
		b22 = B[1][1]
		b23 = B[1][2]
		b31 = B[2][0]
		b32 = B[2][1]
		b33 = B[2][2]

		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		eqs_list = list(eqs)
		# print(eqs_list)
		for j in range(len(eqs_list)):
			eq1 = eqs_list[j]
			x1 = eq1[0][0]
			x2 = eq1[0][1]
			x3 = eq1[0][2]
			y1 = eq1[1][0]
			y2 = eq1[1][1]
			y3 = eq1[1][2]
			df_append = pd.DataFrame([[a11, a12, a13, a21, a22, a23, a31, a32, a33, b11, b12, b13, b21, b22, b23, b31, b32, b33, x1, x2, x3, y1, y2, y3]], columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3'])
			df = df.append(df_append)
	df = df[(~df['x1'].isin([0.0, 1.0]))&(~df['x2'].isin([0.0, 1.0]))&(~df['x3'].isin([0.0, 1.0]))&(~df['y1'].isin([0.0, 1.0]))&(~df['y2'].isin([0.0, 1.0]))&(~df['y3'].isin([0.0, 1.0]))]
	df = df[(df['x1']>1e-10)&(df['x2']>1e-10)&(df['x3']>1e-10)&(df['y1']>1e-10)&(df['y2']>1e-10)&(df['y3']>1e-10)]
	# df = df[(df['y2']<1e-10)]
	df.to_csv(savedata_file, index=False)


def build_3x3game_int(row, col, savedata_file, k):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3'])
	for i in range(500):
		# a1 = np.random.randint(0,10)
		# a2 = np.random.randint(0,10)
		# A[0][0] = a1
		# B[0][0] = a2
		A = np.random.randint(1, 100, (row, col))
		B = np.random.randint(1, 100, (row, col))
		# A[0][0] = 52
		# A[0][1] = 33
		# A[0][2] = 21
		# A[1][0] = 32
		# A[1][1] = 50
		# A[1][2] = 12
		# A[2][0] = 15
		# A[2][1] = 45
		# A[2][2] = 53
		# B[0][0] = k+6
		# B[0][1] = 62
		# B[0][2] = 83
		# B[1][0] = 74
		# B[1][1] = 55
		# B[1][2] = 96
		# B[2][0] = 90
		# B[2][1] = 67
		# B[2][2] = 58

		a11 = A[0][0]
		a12 = A[0][1]
		a13 = A[0][2]
		a21 = A[1][0]
		a22 = A[1][1]
		a23 = A[1][2]
		a31 = A[2][0]
		a32 = A[2][1]
		a33 = A[2][2]

		b11 = B[0][0]
		b12 = B[0][1]
		b13 = B[0][2]
		b21 = B[1][0]
		b22 = B[1][1]
		b23 = B[1][2]
		b31 = B[2][0]
		b32 = B[2][1]
		b33 = B[2][2]

		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		eqs_list = list(eqs)
		# print(eqs_list)
		for j in range(len(eqs_list)):
			eq1 = eqs_list[j]
			x1 = eq1[0][0]
			x2 = eq1[0][1]
			x3 = eq1[0][2]
			y1 = eq1[1][0]
			y2 = eq1[1][1]
			y3 = eq1[1][2]
			df_append = pd.DataFrame([[a11, a12, a13, a21, a22, a23, a31, a32, a33, b11, b12, b13, b21, b22, b23, b31, b32, b33, x1, x2, x3, y1, y2, y3]], columns=['a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33', 'b11', 'b12', 'b13', 'b21', 'b22', 'b23', 'b31', 'b32', 'b33', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3'])
			df = df.append(df_append)
	df = df[(~df['x1'].isin([0.0, 1.0]))&(~df['x2'].isin([0.0, 1.0]))&(~df['x3'].isin([0.0, 1.0]))&(~df['y1'].isin([0.0, 1.0]))&(~df['y2'].isin([0.0, 1.0]))&(~df['y3'].isin([0.0, 1.0]))]
	df = df[(df['x1']>1e-10)&(df['x2']>1e-10)&(df['x3']>1e-10)&(df['y1']>1e-10)&(df['y2']>1e-10)&(df['y3']>1e-10)]
	# df = df[(df['y2']<1e-10)]
	df = df.drop_duplicates()
	df.to_csv(savedata_file, index=False)


def build_2x4game(row, col, savedata_file):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a13', 'a14', 'a21', 'a22', 'a23', 'a24', 'b11', 'b12', 'b13', 'b14', 'b21', 'b22', 'b23', 'b24', 'x1', 'x2', 'y1', 'y2', 'y3', 'y4'])
	for i in range(1000):
		# a1 = np.random.randint(0,10)
		# a2 = np.random.randint(0,10)
		# A[0][0] = a1
		# B[0][0] = a2
		A = np.random.uniform(0.0, 10.0, (row, col))
		B = np.random.uniform(0.0, 10.0, (row, col))
		# A[0][0] = 25
		# A[0][1] = 15
		# A[0][2] = 10
		# A[1][0] = 15
		# A[1][1] = 25
		# A[1][2] = 5
		# A[2][0] = 5
		# A[2][1] = 20
		# A[2][2] = 25
		# B[0][0] = 25
		# B[0][1] = 35
		# B[0][2] = 40
		# B[1][0] = 35
		# # B[1][1] = 25
		# B[1][2] = 45
		# B[2][0] = 45
		# B[2][1] = 30
		# B[2][2] = 25

		a11 = A[0][0]
		a12 = A[0][1]
		a13 = A[0][2]
		a14 = A[0][3]
		a21 = A[1][0]
		a22 = A[1][1]
		a23 = A[1][2]
		a24 = A[1][3]

		b11 = B[0][0]
		b12 = B[0][1]
		b13 = B[0][2]
		b14 = B[0][3]
		b21 = B[1][0]
		b22 = B[1][1]
		b23 = B[1][2]
		b24 = B[1][3]

		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		eqs_list = list(eqs)
		# print(eqs_list)
		for i in range(len(eqs_list)):
			eq1 = eqs_list[i]
			x1 = eq1[0][0]
			x2 = eq1[0][1]
			y1 = eq1[1][0]
			y2 = eq1[1][1]
			y3 = eq1[1][2]
			y4 = eq1[1][3]
			df_append = pd.DataFrame([[a11, a12, a13, a14, a21, a22, a23, a24, b11, b12, b13, b14, b21, b22, b23, b24, x1, x2, y1, y2, y3, y4]], columns=['a11', 'a12', 'a13', 'a14', 'a21', 'a22', 'a23', 'a24', 'b11', 'b12', 'b13', 'b14', 'b21', 'b22', 'b23', 'b24', 'x1', 'x2', 'y1', 'y2', 'y3', 'y4'])
			df = df.append(df_append)
	df = df[(~df['x1'].isin([0.0, 1.0]))&(~df['x2'].isin([0.0, 1.0]))&(~df['y1'].isin([0.0, 1.0]))&(~df['y2'].isin([0.0, 1.0]))&(~df['y3'].isin([0.0, 1.0]))&(~df['y4'].isin([0.0, 1.0]))]
	df = df[(df['x1']>1e-10)&(df['x2']>1e-10)&(df['y1']>1e-10)&(df['y2']>1e-10)&(df['y3']>1e-10)&(df['y3']>1e-10)]
	# df = df[(df['y4']<1e-10)]
	df.to_csv(savedata_file, index=False)


def build_3x4game(row, col, savedata_file):
	# A = np.random.randint(0, 10, (row, col))
	# B = np.random.randint(0, 10, (row, col))
	df = pd.DataFrame(columns=['a11', 'a12', 'a13', 'a14', 'a21', 'a22', 'a23', 'a24', 'a31', 'a32', 'a33', 'a34', 'b11', 'b12', 'b13', 'b14', 'b21', 'b22', 'b23', 'b24', 'b31', 'b32', 'b33', 'b34', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4'])
	for i in range(1000):
		# a1 = np.random.randint(0,10)
		# a2 = np.random.randint(0,10)
		# A[0][0] = a1
		# B[0][0] = a2
		A = np.random.uniform(0.0, 10.0, (row, col))
		B = np.random.uniform(0.0, 10.0, (row, col))

		a11 = A[0][0]
		a12 = A[0][1]
		a13 = A[0][2]
		a14 = A[0][3]
		a21 = A[1][0]
		a22 = A[1][1]
		a23 = A[1][2]
		a24 = A[1][3]
		a31 = A[2][0]
		a32 = A[2][1]
		a33 = A[2][2]
		a34 = A[2][3]

		b11 = B[0][0]
		b12 = B[0][1]
		b13 = B[0][2]
		b14 = B[0][3]
		b21 = B[1][0]
		b22 = B[1][1]
		b23 = B[1][2]
		b24 = B[1][3]
		b31 = B[2][0]
		b32 = B[2][1]
		b33 = B[2][2]
		b34 = B[2][3]

		game = nash.Game(A, B)
		eqs = game.support_enumeration()
		eqs_list = list(eqs)
		# print(eqs_list)
		for i in range(len(eqs_list)):
			eq1 = eqs_list[i]
			x1 = eq1[0][0]
			x2 = eq1[0][1]
			x3 = eq1[0][2]
			y1 = eq1[1][0]
			y2 = eq1[1][1]
			y3 = eq1[1][2]
			y4 = eq1[1][3]
			df_append = pd.DataFrame([[a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34, x1, x2, x3, y1, y2, y3, y4]], columns=['a11', 'a12', 'a13', 'a14', 'a21', 'a22', 'a23', 'a24', 'a31', 'a32', 'a33', 'a34', 'b11', 'b12', 'b13', 'b14', 'b21', 'b22', 'b23', 'b24', 'b31', 'b32', 'b33', 'b34', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4'])
			df = df.append(df_append)
	df = df[(~df['x1'].isin([0.0, 1.0]))&(~df['x2'].isin([0.0, 1.0]))&(~df['x3'].isin([0.0, 1.0]))&(~df['y1'].isin([0.0, 1.0]))&(~df['y2'].isin([0.0, 1.0]))&(~df['y3'].isin([0.0, 1.0]))&(~df['y4'].isin([0.0, 1.0]))]
	df = df[(df['x1']>1e-10)&(df['x2']>1e-10)&(df['y1']>1e-10)&(df['y2']>1e-10)&(df['y3']>1e-10)&(df['y4']>1e-10)]
	# df = df[(df['y4']<1e-10)]
	df.to_csv(savedata_file, index=False)


if __name__ == "__main__":
	# savedata_file = os.path.join("data/", "game2x2_8vals_nopure.csv")
	# build_2x2game(2, 2, savedata_file)

	# savedata_file = os.path.join("data/", "game2x3_1vals_nopure_b11.csv")
	# build_2x3game(2, 3, savedata_file)

	# savedata_file = os.path.join("data/", "game2x4_16vals_nopure.csv")
	# build_2x4game(2, 4, savedata_file)

	# savedata_file = os.path.join("data/", "game3x4_24vals_nopure.csv")
	# build_3x4game(3, 4, savedata_file)

	k = 0
	# savedata_file = os.path.join("data/", "game3x3_3vals_b11-13_c_b21_00.csv")
	# build_3x3game(3, 3, savedata_file, k)

	# savedata_file = os.path.join("data/", "game3x3_9vals_b11-33_int.csv")
	# build_3x3game_int(3, 3, savedata_file, k)

	savedata_file = os.path.join("data/", "game3x3_18vals_int.csv")
	build_3x3game_int(3, 3, savedata_file, k)

	# for k in range(0, 10):
	# 	savedata_file = os.path.join("data/", "game3x3_3vals_b31-33_c_b11_" + str(k) + ".csv")
	# 	# savedata_file = os.path.join("data/", "game3x3_2vals_b11-21_c_b31_" + str(k) + ".csv")
	# 	build_3x3game(3, 3, savedata_file, k)



