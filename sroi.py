import sys

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# sys.path.remove('/home/ejbkdb/catkin_ws/devel/lib/python2.7/dist-packages')

import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__' :

    # Read image
    im = cv2.imread('/home/ejbkdb/anaconda3/envs/py36/4730_final/maze_2.png')

    # Select ROI
    r = cv2.selectROI(im, False)

    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)

    cv2.destroyAllWindows()



img = cv2.imread('/home/ejbkdb/anaconda3/envs/py36/4730_final/maze_2.png',0)
edges = cv2.Canny(img,170,220)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

