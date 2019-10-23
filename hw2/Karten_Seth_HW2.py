# Extra credit question
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

def main(get_input=False):

    points = []
    if get_input == True:
        im = mpimg.imread("hpworld.png")
        plt.imshow(im, cmap="gray")
        plt.axis("off")
        points = plt.ginput(n=4, show_clicks=True)
        print(points)
    else:
        points=[[180.92898624716798, 518.8263345990619],
                [189.199830881649, 242.26996713360347],
                [349.9643734643734, 278.97184019911293],
                [350.48130125402844, 486.776811640448]]
    # Define points in image two (the transformed image)
    points_prime = [[184, 495], #BL
                    [184, 264], #TL
                    [339, 264], #TR
                    [339, 495]] #BR
    A = []
    for p, p_prime in zip(points, points_prime):
        x = p[0]
        y = p[1]
        x_p = p_prime[0]
        y_p = p_prime[1]
        A.append([-x,-y,-1,0,0,0,x_p*x,x_p*y,x_p])
        A.append([0,0,0,-x,-y,-1,y_p*x,y_p*y,y_p])
    A = np.matrix(A)
    u, s, v = np.linalg.svd(A)
    h=v[8]
    h=np.reshape(h,(3,3))
    print("Homography parameters:\n", h)
    im_src = cv2.imread("hpworld.png")
    im_dst = cv2.warpPerspective(im_src, h, (im_src.shape[1],im_src.shape[0]))
    cv2.imshow("Src", im_src)
    cv2.imshow("Dst", im_dst)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(get_input=False)
