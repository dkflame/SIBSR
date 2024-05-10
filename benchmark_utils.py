import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sb

# Load data
def load_data(input_file):
	data = pd.read_csv(input_file)
	return data

def generate_MktRF365():
	input_file = 'dataset/365stocks_noHML.csv'
	data = load_data(input_file)
	data = data.fillna(0)
	MktRF365=data.groupby('date').apply(lambda x: (x['RET']*x['mve']).sum()/x.mve.sum())
	MktRF365=MktRF365.reset_index()
	MktRF365.rename(columns={MktRF365.columns[-1]:'MktRF365'},inplace=True) #重命名一下
	print(MktRF365.head())
	MktRF365.to_csv("dataset/MktRF365.csv")

def duplicate_factor365times():
	input_file = 'dataset/MktRF365.csv'
	data = load_data(input_file)
	Rf = data["MktRF365"] 
	list_MktRF365 = []
	for d in Rf:
	    for i in range(365):
	    	list_MktRF365.append(d)
	print (list_MktRF365)
	dict_MktRF365 = {"MktRF365": list_MktRF365}
	df = pd.DataFrame(dict_MktRF365)
	print(df)
	df.to_csv("dataset/MktRF365_dp.csv")

def remove_duplicates(data):
	data.drop_duplicates(inplace=True)
	return data

def data_clean():
	input_file = os.path.join("benchmark_dso/", "bench_result.csv")
	output_file = os.path.join("benchmark_dso/", "bench_result_clean.csv")
	data = load_data(input_file)
	remove_incomplete_data = data.drop(data[data['success']=='Incomplete'].index)
	result = remove_incomplete_data
	result['experiment_times'] = 1
	# result = remove_duplicates(remove_incomplete_data)
	replacement_map = {
		"[ True]": 1,
		"[True]": 1,
		"[False]": 0
	}
	result["success"].replace(replacement_map, inplace=True)
	result = result.drop(result[result['gamma_similarity']>1.0].index)
	result.sort_values(['benchmark', 'Ho_length', 'Ho'], ascending=[True, True, True], inplace=True)
	result.to_csv(output_file, index=False)

def calculate_mean_by_group():
	input_file = os.path.join("benchmark_dso/", "bench_result_clean.csv")
	output_file = os.path.join("benchmark_dso/", "bench_result_mean_by_group.csv")	
	data = load_data(input_file)
	total_data_amount = len(data)
	result = data.groupby(['benchmark', "groundtruth", "groundtruth_length", 'Ho', 'Ho_length', 'gamma_similarity']).sum()
	result['success'] = result['success']/result['experiment_times']
	# result['DREGS_success'] = 0
	result.sort_values(['benchmark', 'Ho_length', 'Ho'], ascending=[True, True, True], inplace=True)
	result=result.reset_index()
	result.to_csv(output_file, index=False)

	AWGN_1_nums = result[result['benchmark']=='AWGN-1']['experiment_times'].sum()
	Transition_nums = result[result['benchmark']=='Transition']['experiment_times'].sum()
	Z_train1_nums = result[result['benchmark']=='Z-train-1']['experiment_times'].sum()
	R_Distortion_nums = result[result['benchmark']=='R-Distortion']['experiment_times'].sum()
	Entropy_Max_nums = result[result['benchmark']=='Entropy-Max']['experiment_times'].sum()
	return total_data_amount, AWGN_1_nums, Transition_nums, Z_train1_nums, R_Distortion_nums, Entropy_Max_nums

def calculate_mean_by_similarity():
	input_file = os.path.join("benchmark_dso/", "bench_result_mean_by_group.csv")
	output_file = os.path.join("benchmark_dso/", "bench_result_mean_by_similarity.csv")	
	data = load_data(input_file)
	result = data.groupby(['gamma_similarity']).apply(lambda x: (x['success']*x['experiment_times']).sum()/x['experiment_times'].sum())
	result=result.reset_index()
	result.rename(columns={result.columns[-1]:'mean_recovery_rate'},inplace=True) #重命名一下
	# result_DREGS = data.groupby(['gamma_similarity']).apply(lambda x: (x['DREGS_success']*x['experiment_times']).sum()/x['experiment_times'].sum()) # This is not right
	# result_DREGS=result_DREGS.reset_index()
	# result_DREGS.rename(columns={result_DREGS.columns[-1]:'mean_recovery_rate_DREGS'},inplace=True) #重命名一下
	# result = result.join(result_DREGS.set_index('gamma_similarity'), on='gamma_similarity')
	result.to_csv(output_file, index=False)

def plot_scatter_regplot(DREGS_recovery_rate=0, DSR_recovery_rate=0):
	x_dsr = [0,1]
	y_dsr = [DSR_recovery_rate, DSR_recovery_rate]
	plot5 = plt.plot(x_dsr, y_dsr, 'black', ms=1, linestyle='-.', label='DSR')

	x_dregs = [0,1]
	y_dregs_avg = [DREGS_recovery_rate, DREGS_recovery_rate]
	plot4 = plt.plot(x_dregs, y_dregs_avg, 'black', ms=1, linestyle='--', label='DSR-GP')

	input_file = os.path.join("benchmark_dso/", "bench_result_mean_by_group.csv")
	input_file1 = os.path.join("benchmark_dso/", "bench_result_mean_by_similarity.csv")
	data = load_data(input_file)
	data1 = load_data(input_file1)
	df = data[['gamma_similarity', 'success']]
	plot1 = sb.regplot(data=df, x='gamma_similarity', y='success', fit_reg=False, x_jitter=0.003, y_jitter=0.003, scatter_kws={'alpha':1/3}, color='black', label='DSR-GP-SI')

	# df2 = df.copy()
	# df2['success'] = -0.01
	# plot2 = sb.regplot(data=df2, x='gamma_similarity', y='success', fit_reg=False, x_jitter=0.01, y_jitter=0.01, scatter_kws={'alpha':1/3})

	# plot3 = plt.plot(data1['gamma_similarity'], data1['mean_recovery_rate'], 'b-', ms=2)
	f_mean = np.polyfit(data1['gamma_similarity'],data1['mean_recovery_rate'],3)
	yvals_mean = np.polyval(f_mean,data1['gamma_similarity'])
	plot3 = plt.plot(data1['gamma_similarity'],yvals_mean,"black", ms=1, linestyle='-', label='DSR-GP-SI Poly-regression')


	plt.xlabel("similarity", fontsize=14)
	plt.ylabel("recovery rate", fontsize=14)
	plt.legend(bbox_to_anchor=(0.4, 0.8), prop={"size": 8})
	plt.show()


def get_bench_to_repeat():
	input_file = os.path.join("benchmark_dso/", "bench_result_mean_by_group.csv")
	output_file = os.path.join("benchmark_dso/", "bench_result_to_repeat.csv")
	data = load_data(input_file)
	# result = data[(data['success']>0) & (data['success']<1.0)]
	result = data[data['experiment_times']<5]
	result.to_csv(output_file, index=False)


def get_specific_benchs():
	input_file = os.path.join("benchmark_dso/", "bench_result_mean_by_group.csv")
	output_file = os.path.join("benchmark_dso/", "bench_result_specific.csv")
	data = load_data(input_file)
	# result = data[(data['success']>0) & (data['success']<1.0)]
	result = data[(data['experiment_times']<5) & (data['success']>0.9) | (data['experiment_times']==1) & (data['success']<0.1)]
	result.to_csv(output_file, index=False)

def get_DREGS_success():
	input_file = os.path.join("benchmark_dso/", "DREGS_bench_result.csv")
	# output_file = os.path.join("benchmark_dso/", "DREGS_bench_result_clean.csv")
	data = load_data(input_file)
	remove_incomplete_data = data.drop(data[data['success']=='Incomplete'].index)
	result = remove_incomplete_data
	result['experiment_times'] = 1
	# result = remove_duplicates(remove_incomplete_data)
	replacement_map = {
		"[ True]": 1,
		"[True]": 1,
		"[False]": 0
	}
	result["success"].replace(replacement_map, inplace=True)
	result.sort_values('benchmark', ascending=True, inplace=True)
	DREGS_total_success = result["success"].sum()

	DREGS_AWGN_1_success_rate = result[result['benchmark']=='AWGN-1']['success'].sum()/result[result['benchmark']=='AWGN-1']['experiment_times'].sum()
	DREGS_Transition_success_rate = result[result['benchmark']=='Transition']['success'].sum()/result[result['benchmark']=='Transition']['experiment_times'].sum()
	DREGS_Z_train1_success_rate = result[result['benchmark']=='Z-train-1']['success'].sum()/result[result['benchmark']=='Z-train-1']['experiment_times'].sum()
	DREGS_R_Distortion_success_rate = result[result['benchmark']=='R-Distortion']['success'].sum()/result[result['benchmark']=='R-Distortion']['experiment_times'].sum()
	DREGS_Entropy_Max_success_rate = result[result['benchmark']=='Entropy-Max']['success'].sum()/result[result['benchmark']=='Entropy-Max']['experiment_times'].sum()
	return DREGS_total_success, DREGS_AWGN_1_success_rate, DREGS_Transition_success_rate, DREGS_Z_train1_success_rate, DREGS_R_Distortion_success_rate, DREGS_Entropy_Max_success_rate

def get_DSR_success():
	input_file = os.path.join("benchmark_dso/", "DSR_bench_result.csv")
	# output_file = os.path.join("benchmark_dso/", "DREGS_bench_result_clean.csv")
	data = load_data(input_file)
	remove_incomplete_data = data.drop(data[data['success']=='Incomplete'].index)
	result = remove_incomplete_data
	result['experiment_times'] = 1
	# result = remove_duplicates(remove_incomplete_data)
	replacement_map = {
		"[ True]": 1,
		"[True]": 1,
		"[False]": 0
	}
	result["success"].replace(replacement_map, inplace=True)
	result.sort_values('benchmark', ascending=True, inplace=True)
	DSR_total_success = result["success"].sum()

	DSR_AWGN_1_success_rate = result[result['benchmark']=='AWGN-1']['success'].sum()/result[result['benchmark']=='AWGN-1']['experiment_times'].sum()
	DSR_Transition_success_rate = result[result['benchmark']=='Transition']['success'].sum()/result[result['benchmark']=='Transition']['experiment_times'].sum()
	DSR_Z_train1_success_rate = result[result['benchmark']=='Z-train-1']['success'].sum()/result[result['benchmark']=='Z-train-1']['experiment_times'].sum()
	DSR_R_Distortion_success_rate = result[result['benchmark']=='R-Distortion']['success'].sum()/result[result['benchmark']=='R-Distortion']['experiment_times'].sum()
	DSR_Entropy_Max_success_rate = result[result['benchmark']=='Entropy-Max']['success'].sum()/result[result['benchmark']=='Entropy-Max']['experiment_times'].sum()
	return DSR_total_success, DSR_AWGN_1_success_rate, DSR_Transition_success_rate, DSR_Z_train1_success_rate, DSR_R_Distortion_success_rate, DSR_Entropy_Max_success_rate

def main():
	# generate_MktRF365()
	# duplicate_factor365times()

	data_clean()
	total_data_amount, AWGN_1_nums, Transition_nums, Z_train1_nums, R_Distortion_nums, Entropy_Max_nums = calculate_mean_by_group()
	calculate_mean_by_similarity()

	DREGS_total_success, DREGS_AWGN_1_success_rate, DREGS_Transition_success_rate, DREGS_Z_train1_success_rate, DREGS_R_Distortion_success_rate, DREGS_Entropy_Max_success_rate = get_DREGS_success()
	DREGS_recovery_rate = DREGS_total_success/total_data_amount

	DSR_total_success, DSR_AWGN_1_success_rate, DSR_Transition_success_rate, DSR_Z_train1_success_rate, DSR_R_Distortion_success_rate, DSR_Entropy_Max_success_rate = get_DSR_success()
	DSR_recovery_rate = DSR_total_success/total_data_amount

	print(DREGS_recovery_rate)
	print(DSR_recovery_rate)
	
	plot_scatter_regplot(DREGS_recovery_rate, DSR_recovery_rate)

	# print("total_data_amount: " + str(total_data_amount) + ", DREGS_total_success: " + str(DREGS_total_success))
	# print("total_data_amount: " + str(total_data_amount) + ", DSR_total_success: " + str(DSR_total_success))
	# print ('AWGN-1 success rate :' + str(DREGS_AWGN_1_success_rate))
	# print ('Transition success rate :' + str(DREGS_Transition_success_rate))
	# print ('Z-train-1 success rate :' + str(DREGS_Z_train1_success_rate))
	# print ('R-Distortion success rate :' + str(DREGS_R_Distortion_success_rate))
	# print ('Entropy-Max success rate :' + str(DREGS_Entropy_Max_success_rate))

	

if __name__ == "__main__":

	main()
	# get_bench_to_repeat()
	# get_specific_benchs()

