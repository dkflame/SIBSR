from dso import DeepSymbolicOptimizer
from dso import DeepSymbolicRegressor
import pandas as pd
import numpy as np
import os
import multiprocessing
import commentjson as json
from copy import deepcopy
import random
import pandas as pd
import numpy as np


def load_data(input_file):
	data = pd.read_csv(input_file)
	return data


def main():
	input_file = "bench_result_specific.csv"
	# input_file = "bench_result_to_repeat.csv"
	data = load_data(input_file)
	for benchmark, Ho_str in zip(data['benchmark'], data['Ho']):

		# Create and train the model
		config_json = os.path.join("../config/", "benchmark_regression.json")
		new_config_json = os.path.join("../config/benchmark_regression-" + benchmark + ".json")

		Ho_str = Ho_str.strip("[")
		Ho_str = Ho_str.strip("]")
		Ho_str = Ho_str.replace("'", "")
		Ho_str = Ho_str.replace(" ", "")
		print(Ho_str)
		Ho_traversal = Ho_str.split(",")

		with open(config_json, encoding='utf-8') as f:
			json_data = json.load(f)
			json_data["task"]["dataset"]["name"] = benchmark
			json_data["training"]["Ho"] = Ho_traversal

		with open(new_config_json, 'w') as f:
		    f.write(json.dumps(json_data))

		print(Ho_traversal)
		model = DeepSymbolicOptimizer(new_config_json)
		model.train()

		# try:
		# 	model = DeepSymbolicOptimizer(new_config_json)
		# 	model.train()
		# except:
		#     print("Invalid Ho: ")
		#     print(Ho_traversal)

if __name__ == "__main__":

	for i in range(2):
		main()

