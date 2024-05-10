import numpy as np
import ffx
import pandas as pd

# Load data
def load_data(input_file):
	data = pd.read_csv(input_file)
	# data = data.fillna(0)
	print(data.head())
	return data

variables = ["b11", "b12","b13", "b21", "b22", "b23", "b31", "b32", "b33"]
target = "x1"

input_file = 'data/game3x3_18vals_int.csv'
data = load_data(input_file)
rows, cols = data.shape
train_data_rows = int(rows * 0.8)
test_data_rows = int(rows * 0.2)

training_data = data.head(train_data_rows)
testing_data = data.tail(test_data_rows)

train_y = training_data[target].to_numpy()
train_X = training_data[variables].to_numpy()
test_y = testing_data[target].to_numpy()
test_X = testing_data[variables].to_numpy()



# train_X = np.array( [ (1.5,2,3), (4,5,6), (7,8,9) ] )
# train_y = np.array( [1,2,3])

# test_X = np.array( [ (5.241,1.23, 3.125), (1.1,0.124,0.391), (3.2,0.82,0.71) ] )
# test_y = np.array( [3.03,0.9113,1.823])

models = ffx.run(train_X, train_y, test_X, test_y, variables, verbose=True)
for model in models:
	yhat = model.simulate(test_X)
	print(model)