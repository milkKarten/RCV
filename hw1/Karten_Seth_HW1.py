# Author: Seth Karten
# Note: Use python3 with numpy installed
import numpy as np

def main():
    A = np.matrix([ [4.29, 2.2, 5.51],
                    [5.20, 10.1, -8.24],
                    [1.33, 4.8, -6.62]
                    ])
    _svd = np.linalg.svd(A)
    print("Matrix:")
    print("%lf\t%lf\t%lf" % (4.29, 2.2, 5.51))
    print("%lf\t%lf\t%lf" % (5.20, 10.1, -8.24))
    print("%lf\t%lf\t%lf" % (1.33, 4.8, -6.62))
    print("The singular values are %lf, %lf, and %lf." % (_svd[1][0], _svd[1][1], _svd[1][2]))
    print("The rank of the matrix is %d." % np.linalg.matrix_rank(A))
    return

if __name__ == "__main__":
    main()
