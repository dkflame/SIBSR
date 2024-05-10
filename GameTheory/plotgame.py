import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from train_game_dso import load_data


def plot_curve(data):
	# x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	# y1=[14.02, 80.58, 98.5, 161.50, 193.94, 223.13, 245.42, 257.57, 276.49, 273.56, 282.21]
	# y2=[14.01, 70.33, 107.28, 188.69, 217.17, 239.14, 268.46, 274.17, 278.81, 280.80, 281.34]

	data = data.sort_values(by=['b11'])
	x = data["b11"]
	y1 = data["x1"]
	# y2 = data["x1_pred"]

	plot1 = plt.plot(x, y1, 'b-', ms=2)
	# plot2 = plt.plot(x, y2, 'r-', ms=2)

	## scatter
	plot1 = plt.scatter(x,y1)

	## curve
	# f1 = np.polyfit(x,y1,6)
	# yvals1 = np.polyval(f1,x)
	# plot1 = plt.plot(x,yvals1,"b")
	# f2 = np.polyfit(x,y2,6)
	# yvals2 = np.polyval(f2,x)
	# plot2 = plt.plot(x,yvals2,"r")

	plt.xlabel("b11")
	plt.ylabel("x1")

	plt.show()



if __name__ == "__main__":
	input_file = 'data/game3x3_1vals_nopure_b11.csv'
	data = load_data(input_file)
	plot_curve(data)
	# plot_bar()

