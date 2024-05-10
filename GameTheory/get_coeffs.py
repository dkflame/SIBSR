import numpy as np
import pandas as pd
import os


def get_denominator_coeff(const_list, save_file):
	coeff = 0
	df = pd.DataFrame()
	for i in range(len(const_list)):
		df1 = df.copy()
		list1 = [const_list[i]]*np.power(2, i)
		list2 = [(-1)*const_list[i]]*np.power(2, i)
		df[i] = list1
		df1[i] = list2
		df = pd.concat([df, df1], ignore_index=True)
	df['col_sum'] = df.apply(lambda x: x.sum(), axis=1)
	df.to_csv(save_file, index=False)
	
def get_numerator_coeff(const_lists, save_file):
	all_df = pd.DataFrame()
	for const_list in const_lists:
		coeff = 0
		df = pd.DataFrame()
		for i in range(len(const_list)):
			df1 = df.copy()
			list1 = [const_list[i]]*np.power(2, i)
			list2 = [(-1)*const_list[i]]*np.power(2, i)
			df[i] = list1
			df1[i] = list2
			df = pd.concat([df, df1], ignore_index=True)
		df['col_sum'] = df.apply(lambda x: x.sum(), axis=1)
		all_df = pd.concat([all_df, df], ignore_index=True)
	all_df.to_csv(save_file, index=False)

def get_3_payoffs(const_list):
	const_lists = []
	for i in range(len(const_list)-2):
		for j in range(i+1,len(const_list)-1):
			for k in range(j+1,len(const_list)):
				payoff_list = [const_list[i], const_list[j], const_list[k]]
				const_lists.append(payoff_list)
	print(const_lists)
	return const_lists

def get_4_payoffs(const_list):
	const_lists = []
	for i in range(len(const_list)-3):
		for j in range(i+1,len(const_list)-2):
			for k in range(j+1,len(const_list)-1):
				for l in range(k+1,len(const_list)):
					payoff_list = [const_list[i], const_list[j], const_list[k], const_list[l]]
					const_lists.append(payoff_list)
	print(const_lists)
	return const_lists

if __name__ == "__main__":
	const_list = [7.2,8.3,7.4,5.5,9.6,9.0,6.7,5.8]
	denominator_file = os.path.join("data/", "game3x3_1vals_nopure_b11_denominator_coeff.csv")
	# get_denominator_coeff(const_list, denominator_file)
	# const_lists = get_3_payoffs(const_list)
	# numerator_3payoff_file = os.path.join("data/", "game3x3_1vals_nopure_b11_numerator_3payoff_coeff.csv")
	# get_numerator_coeff(const_lists, numerator_3payoff_file)
	const_lists = get_4_payoffs(const_list)
	numerator_4payoff_file = os.path.join("data/", "game3x3_1vals_nopure_b11_numerator_4payoff_coeff.csv")
	get_numerator_coeff(const_lists, numerator_4payoff_file)



