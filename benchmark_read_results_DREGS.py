import os
import pandas as pd
import numpy as np
import json

def load_folders(folder):
	folder_walk = os.walk(folder)
	all_bench_results = []
	for path, dir_list, file_list in folder_walk:
		for dir_name in dir_list:
			bench_folder = os.path.join(path, dir_name)
			print(bench_folder)
			bench_result = load_files(bench_folder)
			all_bench_results.append(bench_result)
	all_bench_results_df = pd.DataFrame(all_bench_results, columns=['benchmark', 'success'])
	return all_bench_results_df

def load_files(folder):
	folder_walk = os.walk(folder)
	bench_result = []
	success = None
	for path, dir_list, file_list in folder_walk:		
		for file_name in file_list:
			bench_file = os.path.join(path, file_name)
			print(bench_file)
			if file_name == "config.json":
				benchmark = read_json(bench_file)
				bench_result.append(benchmark)

			if "pf.csv" in file_name:
				result_csv = pd.read_csv(bench_file)
				success = result_csv.tail(1)["success"].values
	if success is not None:
		bench_result.append(success)
	if len(bench_result) == 1:
		bench_result.append('Incomplete')
	return bench_result


def read_json(json_file):
	with open(json_file, encoding='utf-8') as f:
		json_data = json.load(f)
		benchmark = json_data["task"]["dataset"]["name"]
	return benchmark


if __name__ == "__main__":
	# folder = "benchmark_dso/DREGS_bench"
	# result_csv = os.path.join("benchmark_dso/", "DREGS_bench_result.csv")

	folder = "benchmark_dso/DSR_bench"
	result_csv = os.path.join("benchmark_dso/", "DSR_bench_result.csv")

	all_bench_results_df = load_folders(folder)
	print(all_bench_results_df)
	all_bench_results_df.to_csv(result_csv,index=False)


