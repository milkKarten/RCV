# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy import cos, sin, pi
import sys

def cosd(x):
    return cos(x*pi/180)

def sind(x):
    return sin(x*pi/180)

def matrix_to_string(m):
    s = ""
    for i in range(len(m)-1):
        for j in range(m.shape[1]):
            if (j != m.shape[1]-1):
                s += str(m[i,j]) + "\t"
            else:
                s += str(m[i,j])
        if i != len(m) - 2:
            s += "\n"
    return s

def main():
    try:
        theta1 = float(input("Enter first angle value: "))
        theta2 = float(input("Enter second angle value: "))
        theta3 = float(input("Enter third angle value: "))
    except:
        print("Bad user input")
        exit(0)
    L1 = 10
    L2 = 10
    L3 = 3
    T1 = np.matrix([[cosd(theta1), -sind(theta1), 0, L1*cosd(theta1)],
                    [sind(theta1), cosd(theta1), 0, L1*sind(theta1)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    T2 = np.matrix([[cosd(theta2), -sind(theta2), 0, L2*cosd(theta2)],
                    [sind(theta2), cosd(theta2), 0, L2*sind(theta2)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    T3 = np.matrix([[cosd(theta3), -sind(theta3), 0, L3*cosd(theta3)],
                    [sind(theta3), cosd(theta3), 0, L3*sind(theta3)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    print("0T1:\n", matrix_to_string(T1))
    print("1T2:\n", matrix_to_string(T2))
    print("2T3:\n", matrix_to_string(T3))
    T03 = T1 * T2 * T3
    print("0T3:\n", matrix_to_string(T03))
    print("The location of the wrist with respect to the reference coordinate frame is...\nx = %f\ty = %f" % (T03[0,3], T03[1,3]))
    return

if __name__ == "__main__":
    main()
