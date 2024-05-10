import distance

def normal_leven(list1, list2):
	len_list1 = len(list1) + 1
	len_list2 = len(list2) + 1
	matrix = [0 for n in range(len_list1 * len_list2)]
	for i in range(len_list1):
		matrix[i] = i
	for j in range(0, len(matrix), len_list1):
		if j % len_list1 == 0:
			matrix[j] = j // len_list1
	for i in range(1, len_list1):
		for j in range(1, len_list2):
			if list1[i-1] == list2[j-1]:
				cost = 0
			else:
				cost = 1
			matrix[j*len_list1+i] = min(matrix[(j-1)*len_list1+i]+1, matrix[j*len_list1+(i-1)]+1, matrix[(j-1)*len_list1+(i-1)] + cost)
	return matrix[-1]

def calculate_ss():

	T_string = "add,mul,div,1.0,2.0,mul,div,sqrt,mul,2.0,3.14159,exp,neg,div,n2,x1,2.0,mul,div,1.0,2.0,mul,div,sqrt,mul,2.0,3.14159,exp,neg,div,n2,x2,2.0"
	H_string = "div,sqrt,div,sqrt,1.0,mul,1.0,1.0,sqrt,exp,n2,x1"
	T = T_string.split(',')
	H = H_string.split(',')

	ss = normal_leven(T, H)/(len(T)+len(H))
	print(len(T))
	print(len(H))
	print(normal_leven(T, H))
	print(ss)


if __name__ == "__main__":
	calculate_ss()