import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random

def plot_curve():
	x=[0, 0.08, 0.12, 0.12, 0.14, 0.15, 0.18, 0.2, 0.22, 0.23, 0.27, 0.31, 0.54, 0.56, 0.6, 0.7, 0.8, 0.9, 1]
	y1=[1, 1, 0.93, 0.93, 0.79, 0.71, 0.64, 0.57, 0.5, 0.43, 0.36, 0.29, 0.21, 0.14, 0.07, 0.003, 0.003, 0.003, 0.003]
	y2=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	plot1 = plt.plot(x, y1, 'b-', ms=2)
	plot2 = plt.plot(x, y2, 'r-', ms=1)

	# f1 = np.polyfit(x,y1,3)
	# yvals1 = np.polyval(f1,x)
	# plot1 = plt.plot(x,yvals1,"b")

	# f2 = np.polyfit(x,y2,6)
	# yvals2 = np.polyval(f2,x)
	# plot2 = plt.plot(x,yvals2,"r")

	plt.xlabel("ss value")
	plt.ylabel("recovery rate")
	# plt.ylabel("runtime (sec)")

	plt.legend(['SIBSR', 'SIBSR-base'], loc='upper right')

	plt.show()

def plot_bar():
	nguyen_result = [82.83, 86, 88.67, 91.42, 100]
	livermore_result = [30.41, 56.36, 58.18, 71.09, 84.95]
	data_result = np.array([nguyen_result, livermore_result])
	data = pd.DataFrame(data_result)
	data.columns = ['DSR', 'GEGL', 'Random Restart GP', 'DREGS', 'SIBSR(ours)']
	data.index = ['Ng', 'Lv']
	print(data)

	plot = data.plot(kind='bar',
		figsize=(10,5),
		title='Recovery Rate Comparison on Different Benchmarks')
	plt.xlabel("benchmark")
	plt.ylabel("recovery rate")
	plt.show()


def plot_sin():
	x = np.random.random(500)
	x_sort = np.sort(x)
	y = np.sin(2*np.pi*x_sort) +2 
	gauss_noise(x_sort, y)

	plot1 = plt.plot(x_sort, y, 'x', ms=2)
	plt.show()

def plot_fun(): 
	x = np.random.random(1000)
	# x2 = -5*np.random.random(1000)
	# x = x+x2
	x = np.sort(x)
	# b11 = 2.8
	# b12 = 1.6
	# b21 = x
	# b22 = 9.4
	# y = 1/np.sqrt(2*np.pi)*np.exp(-x*x/2)
	# y = -x*np.log(x)-(1-x)*np.log(1-x)
	# y = np.exp(-x)/np.square(1+np.exp(-x))
	# y = np.sin(x*x)*np.cos(x)-5
	y = np.log(x+1) + np.log(x*x+1)
	# y = (b22 - b21)/(b11 - b12 - b21 + b22)
	plot1 = plt.plot(x, y, 'b-', ms=2)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()
	df = pd.DataFrame(columns=('x', 'y'))
	df['x'] = x
	df['y'] = y
	print(df)


def plot_3d_fun():
	fig = plt.figure()
	ax3 = plt.axes(projection='3d')

	xx = np.arange(-5, 5, 0.5)
	yy = np.arange(-5, 5, 0.5)
	X, Y = np.meshgrid(xx, yy)
	Z = 6*np.sin(X)*np.cos(Y)
	# Z = np.power(X, 4) - np.power(X, 3) + 0.5*np.power(Y, 2) - Y

	ax3.plot_surface(X, Y, Z, cmap='rainbow')
	ax3.set_xlabel('X axis')
	ax3.set_ylabel('Y axis')
	ax3.set_zlabel('Z axis')


	plt.show()

def gauss_noise(x, y):
	mu = 0
	sigma = 0.005
	for i in range(len(x)):
		x[i] += random.gauss(mu, sigma)
		y[i] += random.gauss(mu, sigma)


if __name__ == "__main__":
	plot_curve()
	# plot_bar()
	# plot_sin()
	# plot_fun()
	# plot_3d_fun()

