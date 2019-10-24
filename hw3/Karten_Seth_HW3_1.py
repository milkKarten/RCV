# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def drawmyobject(views, axes):
    for view, axis in zip(views, axes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for line in view:
            ax.add_line(line)
        ax.axis(axis)
        fig.canvas.mpl_connect('key_press_event', key_callback)
        plt.show()
    return

def key_callback(event):
    if event.key == 'n':
        plt.close()
    elif event.key == 'q':
        exit(0)


'''
    Pass R, T in world frame
'''
def rotate_and_get_lines(R, t, views, axes):
    # Load simple house
    Ps = [[0-0.5,0-0.5,0-0.5],#1: 2,3,4
         [1-0.5,0-0.5,0-0.5],#2: 5,6
         [0-0.5,1-0.5,0-0.5],#3: 5,7
         [0-0.5,0-0.5,1-0.5],#4: 6,7,9
         [1-0.5,1-0.5,0-0.5],#5: 8
         [1-0.5,0-0.5,1-0.5],#6: 8,9
         [0-0.5,1-0.5,1-0.5],#7: 8,9
         [1-0.5,1-0.5,1-0.5],#8: 9
         [0.5-0.5,0.5-0.5,1.5-0.5]]  #9:
    K = np.matrix([ [-100, 0, 200],
                    [0, -100, 200],
                    [0, 0, 1]])
    M = K * R.T
    M_t = K * -R.T*t
    M = np.c_[M, M_t]
    print(M)
    exit()
    p = []
    for P in Ps:
        P.append(1)
        P = np.matrix(P)
        print(M, P.T)
        p.append(M*P.T)
    lines = []
    out_x = [0]
    out_y = [0]
    min_x = min_y = max_x = max_y = -1000000
    for point, old_p in zip(p, Ps):
        z = point[2]
        if z == 0:
            z = 1
        x_new = point[0]/z
        y_new = point[1]/z
        if min_x == -1000000:
            min_x = max_x = x_new[0,0]
            min_y = max_y = y_new[0,0]
        min_x = min(min_x, x_new[0,0])
        max_x = max(max_x, x_new[0,0])
        min_y = min(min_y, y_new[0,0])
        max_y = max(max_y, y_new[0,0])
        # print(old_p, x_new[0,0], y_new[0,0])
        out_x.append(x_new[0,0])
        out_y.append(y_new[0,0])
    axis = [min_x-2, max_x+2, min_y-2, max_y+2]
    lines.append(Line2D([out_x[1],out_x[2]], [out_y[1],out_y[2]]))
    lines.append(Line2D([out_x[1],out_x[3]], [out_y[1],out_y[3]]))
    lines.append(Line2D([out_x[1],out_x[4]], [out_y[1],out_y[4]]))
    lines.append(Line2D([out_x[2],out_x[5]], [out_y[2],out_y[5]]))
    lines.append(Line2D([out_x[2],out_x[6]], [out_y[2],out_y[6]]))
    lines.append(Line2D([out_x[3],out_x[5]], [out_y[3],out_y[5]]))
    lines.append(Line2D([out_x[3],out_x[7]], [out_y[3],out_y[7]]))
    lines.append(Line2D([out_x[4],out_x[6]], [out_y[4],out_y[6]]))
    lines.append(Line2D([out_x[4],out_x[7]], [out_y[4],out_y[7]]))
    lines.append(Line2D([out_x[4],out_x[9]], [out_y[4],out_y[9]]))
    lines.append(Line2D([out_x[5],out_x[8]], [out_y[5],out_y[8]]))
    lines.append(Line2D([out_x[6],out_x[8]], [out_y[6],out_y[8]]))
    lines.append(Line2D([out_x[6],out_x[9]], [out_y[6],out_y[9]]))
    lines.append(Line2D([out_x[7],out_x[8]], [out_y[7],out_y[8]]))
    lines.append(Line2D([out_x[7],out_x[9]], [out_y[7],out_y[9]]))
    lines.append(Line2D([out_x[8],out_x[9]], [out_y[8],out_y[9]]))
    views.append(lines)
    axes.append(axis)
    return

def main():
    views = []
    axes = []
    R = np.matrix([[0,0,-1],[1,0,0],[0,-1,0]])
    t = np.matrix([10,0,0]).T
    rotate_and_get_lines(R, t, views, axes)
    exit()
    R2 = np.matrix([[-1.0/np.sqrt(2),0,0],[1.0/np.sqrt(2),0,-1.0/np.sqrt(2)],[0,-1,0]])
    t2 = np.matrix([np.sqrt(50),np.sqrt(50),0]).T
    rotate_and_get_lines(R2, t2, views, axes)
    R3 = np.matrix([[-1,0,0],[0,0,-1],[0,-1,0]])
    t3 = np.matrix([0,10,0]).T
    rotate_and_get_lines(R3, t3, views, axes)
    R4 = np.matrix([[0,0,1],[-1,0,0],[0,-1,0]])
    t4 = np.matrix([-10,0,0]).T
    rotate_and_get_lines(R4, t4, views, axes)
    R5 = np.matrix([[1,0,0],[0,0,1],[0,-1,0]])
    t5 = np.matrix([0,-10,0]).T
    rotate_and_get_lines(R5, t5, views, axes)
    drawmyobject(views, axes)
    return

if __name__ == "__main__":
    main()
