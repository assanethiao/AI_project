#Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt

def coef_estimation( x,y ):
    n = np.size(x)
    mean_x, mean_y= np.mean(x), np.mean(y)
    cross_xy = np.sum(y*x) - n*mean_x*mean_y
    cross_xx = np.sum(x*x) - n*mean_x*mean_x
    reg_b_1 = cross_xy/cross_xx
    reg_b_0 = mean_y - reg_b_1*mean_x
    return reg_b_0, reg_b_1

def plot_regression_line( x,y, b):
    plt.scatter(x,y, color = 'red', marker = 'o', s = 70)
    y_pred = b[0] + b[1]*x
    plt.plot(x,y_pred, color = 'blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def main():
    x = np.random.rand(50)
    y = 3 * x + 2 + np.random.randn(50) * 0.1
    b = coef_estimation(x,y)
    print("Estimated coefficients:\nreg_b_0 = {} \nreg_b_1 ={}".format(b[0],b[1]))
    plot_regression_line(x,y,b)

if __name__ == "__main__":
        main()

