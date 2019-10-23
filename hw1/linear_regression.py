import numpy as np
from matplotlib import pyplot as plt

def main():
    x1 = 0.; y1 = 1.
    x2 = 1.; y2 = 3.2
    x3 = 1.9; y3 = 5.
    x4 = 3.; y4 = 7.2
    x5 = 3.9; y5 = 9.3
    x6 = 5.; y6 = 11.1
    A = np.matrix( [[1,x1],
                    [1,x2],
                    [1,x3],
                    [1,x4],
                    [1,x5],
                    [1,x6]
                    ])
    b = np.matrix([y1,y2,y3,y4,y5,y6]).T
    R = np.linalg.inv(A.T*A)*A.T*b
    print(R)
    X = [x1,x2,x3,x4,x5,x6]
    Y = [y1,y2,y3,y4,y5,y6]
    X1 = np.linspace(0,5,100)
    Y1 = np.array([X1 * R[1,0] + R[0,0]]).T
    plt.plot(X,Y)
    plt.plot(X1,Y1)
    plt.show()
    return

if __name__ == "__main__":
    main()
