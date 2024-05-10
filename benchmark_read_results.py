import os
import pandas as pd
import numpy as np
import json

def get_groundtruth(benchmark):
	groundtruth_str = ""
	if benchmark == "AWGN-2":	
		groundtruth_str = "mul,div,1.0,2.0,div,log,add,1.0,div,x1,n2,x2,log,2.0"
	if benchmark == "AWGN-3":	
		groundtruth_str = "mul,x1,log,add,1.0,div,x2,mul,x1,x3"
	if benchmark == "AWGN-4":	
		groundtruth_str = "mul,x1,div,log,add,1.0,div,mul,x2,x4,mul,x1,x3,log,2.0"
	if benchmark == "D-Entropy":	
		groundtruth_str = "mul,div,1.0,2.0,add,mul,x1,log,mul,mul,2.0,.3.14159,exp,1.0,log,x2"
	if benchmark == "Entropy":	
		groundtruth_str = "sub,neg,div,mul,x1,log,x1,log,2.0,div,mul,div,1.0,x1,log,sub,1.0,x1,log,2.0"
	if benchmark == "Folded-N-Distribution":	
		groundtruth_str = "add,mul,div,1.0,2.0,mul,div,sqrt,mul,2.0,3.14159,exp,neg,div,n2,x1,2.0,mul,div,1.0,2.0,mul,div,sqrt,mul,2.0,3.14159,exp,neg,div,n2,x2,2.0"
	if benchmark == "LDPC-3":	
		groundtruth_str = "div,1.0,add,1.0,exp,neg,div,mul,mul,2.0,x1,x2,n2,x3"
	if benchmark == "LDPC-4":	
		groundtruth_str = "log,div,add,1.0,exp,add,x1,x2,add,exp,x1,exp,x2"
	if benchmark == "Logistic-Distribution":	
		groundtruth_str = "div,exp,neg,x1,n2,add,1.0,exp,neg,1.0"
	if benchmark == "N-Distribution":	
		groundtruth_str = "mul,div,1.0,sqrt,mul,2.0,3.14159,exp,neg,div,n2,x1,2.0"
	if benchmark == "RMSE-3":	
		groundtruth_str = "sqrt,add,add,n2,sub,x1,1.0,n2,sub,x2,2.0,n2,sub,x3,3.0"
	if benchmark == "SNR":	
		groundtruth_str = "mul,10.0,div,log,div,n2,x1,n2,x2,log,10.0"
	if benchmark == "Z-train-2":	
		groundtruth_str = "div,mul,2.0,div,1.0,x1,n2,sub,1.0,div,1.0,x1"
	if benchmark == "Z-train-3":	
		groundtruth_str = "div,mul,sin,1.0,div,1.0,x1,add,sub,1.0,mul,2.0,mul,cos,1.0,div,1.0,x1,div,1.0,n2,x1"
	if benchmark == "Z-train-4":	
		groundtruth_str = "div,mul,cos,1.0,div,1.0,x1,add,sub,1.0,mul,2.0,mul,cos,1.0,div,1.0,x1,div,1.0,n2,x1"
	if benchmark == "AWGN-1":	
		groundtruth_str = "mul,div,1.0,2.0,log,add,1.0,div,x1,x2"
	if benchmark == "Transition":	
		groundtruth_str = "div,1.0,add,1.0,n2,div,mul,2.0,x1,1.0"
	if benchmark == "Z-train-1":	
		groundtruth_str = "div,x1,sub,x1,1.0"
	if benchmark == "R-Distortion":	
		groundtruth_str = "mul,div,1.0,2.0,log,div,n2,x1,x2"
	if benchmark == "Entropy-Max":	
		groundtruth_str = "mul,div,1.0,x2,exp,div,neg,x1,x2"
	groundtruth_list = groundtruth_str.split(",")
	return groundtruth_list

def load_folders(folder):
	folder_walk = os.walk(folder)
	all_bench_results = []
	for path, dir_list, file_list in folder_walk:
		for dir_name in dir_list:
			bench_folder = os.path.join(path, dir_name)
			print(bench_folder)
			bench_result = load_files(bench_folder)
			all_bench_results.append(bench_result)
	all_bench_results_df = pd.DataFrame(all_bench_results, columns=['benchmark', "groundtruth", "groundtruth_length", 'Ho', 'Ho_length', 'gamma_similarity', 'success'])
	return all_bench_results_df

def load_files(folder):
	folder_walk = os.walk(folder)
	# result_pd = pd.DataFrame(columns=['benchmark', 'Ho', 'success'])
	bench_result = []
	success = None
	for path, dir_list, file_list in folder_walk:		
		for file_name in file_list:
			bench_file = os.path.join(path, file_name)
			print(bench_file)
			if file_name == "config.json":
				benchmark, Ho = read_json(bench_file)
				bench_result.append(benchmark)
				benchmark_truth = get_groundtruth(benchmark)
				bench_result.append(benchmark_truth)
				bench_result.append(len(benchmark_truth))
				bench_result.append(Ho)
				bench_result.append(len(Ho))
				bench_result.append(len(Ho)/len(benchmark_truth))

			if "pf.csv" in file_name:
				result_csv = pd.read_csv(bench_file)
				success = result_csv.tail(1)["success"].values
	if success is not None:
		bench_result.append(success)
	if len(bench_result) == 6:
		bench_result.append('Incomplete')
	return bench_result


def read_json(json_file):
	with open(json_file, encoding='utf-8') as f:
		json_data = json.load(f)
		benchmark = json_data["task"]["dataset"]["name"]
		Ho = json_data["training"]["Ho"]
	return benchmark, Ho


if __name__ == "__main__":
	folder = "benchmark_dso/log"
	result_csv = os.path.join("benchmark_dso/", "bench_result.csv")
	all_bench_results_df = load_folders(folder)
	print(all_bench_results_df)
	all_bench_results_df.to_csv(result_csv,index=False)


