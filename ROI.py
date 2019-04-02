# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# sys.path.remove('/home/ejbkdb/catkin_ws/devel/lib/python2.7/dist-packages')

import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

def navigable(current_Frame):
    frame = current_Frame
    # try:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower_thresh = 150
    upper_thresh = 255
    mask = cv2.inRange(gray, lower_thresh, upper_thresh)
    maskedImg = cv2.bitwise_and(gray, mask)
    # maskedImg[maskedImg == 0] = 1
    # maskedImg[maskedImg > 1] = 0

    return maskedImg

def findedges(maskedImg):
    edges = cv2.Canny(maskedImg,10,200)
    return edges

def gray2red(gray):
    rgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    rgb[:, :, 0] = rgb[:, :, 0] *0
    rgb[:, :, 1] = rgb[:, :, 1] *0
    rgb[:, :, 2] = rgb[:, :, 2]
    return rgb
def gray2BG(img):
    rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    rgb[:, :, 0] = rgb[:, :, 0]
    rgb[:, :, 1] = rgb[:, :, 1]
    rgb[:, :, 2] = rgb[:, :, 2]*0
    return rgb

def navto(direction):

    for i in range(0,len(direction)):
        if i == len(direction)-1:
            return i
        if direction[i] == 0 and direction[i+1] == 1:
            return i

def moveoptions(test):
    test = cv2.cvtColor(nav, cv2.COLOR_BGR2GRAY)
    test[test > 0] = 1
    x = int(test.shape[1] / 2)
    y = int(test.shape[0] / 2)

    down = test[y:test.shape[0], x]
    up = test[0:y, x]
    left = test[y, 0:x]
    right = test[y, x:test.shape[1]]

    dist_down = navto(down)
    dist_up = navto(up)
    dist_left = navto(left)
    dist_right = navto(right)

    pos_down = (y + dist_down, x)
    pos_up = (y - dist_up, x)
    pos_left = (y, x - dist_left)
    pos_right = (y, x - dist_right)
    # move = [pos_down, pos_up, pos_left, pos_right]
    move = [dist_down,dist_up,dist_left,dist_right]
    return move

def nextmove(move,loc,target=None):
    if target == None:
        print('notworking')
        maxmove = move.index(max(move))
        if maxmove == 0:
            return (loc[0],loc[1]+max(move))
        if maxmove == 1:
            return (loc[0], loc[1]-max(move))
        if maxmove == 2:
            return (loc[0]-max(move), loc[1])
        if maxmove == 3:
            return (loc[0]+max(move), loc[1])

    if target != None:
        down = (loc[0], loc[1]+move[0])
        up = (loc[0], loc[1]-move[1])
        left = (loc[0]-move[2], loc[1])
        right = (loc[0]+move[3], loc[1])
        moves = [down,up,left,right]
        q = []
        for i in range(0,len(moves)):
            x = target[1]
            y = target[0]
            q.append((np.array((moves[i][0]-x)**2 + (moves[i][1]-y)**2)**(1/2)))
        mindist = q.index(min(q))
        if mindist == 0:
            # return (loc[0], loc[1]+move[0])
            return down
        if mindist == 1:
            # return (loc[0], loc[1]-move[1])
            return up
        if mindist == 2:
            # return (loc[0]-move[2], loc[1])
            return left
        if mindist == 3:
            # return (loc[0]+move[3], loc[1]
            return right



(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[0]

    # if int(major_ver) > 3:
    #     tracker = cv2.Tracker_create(tracker_type)
    # else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Read video
    # video = cv2.VideoCapture("/home/ejbkdb/anaconda3/envs/py36/4730_final/output2.avi")
    video = cv2.VideoCapture(0)

    width_height = (int(video.get(4)), int(video.get(3)),3)
    worldmap = np.zeros(width_height)
    worldmap2 = np.zeros(width_height)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    # x= int(340)
    # y=int(90)
    # w = int(33)
    # bbox = (x,y,w,w)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame)
    #####################
    ## Select boxes

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects

    #####################################
    w = int(video.get(4))
    h = int(video.get(3))
    d = int(3)
    # framecount = int(437)
    # framearray = np.empty((framecount, w, h, d), dtype=frame.dtype)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    path = []
    history = []
    z= 0
    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            findme = (200,200)

            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            loc = int(((p1[0] + p2[0]) / 2)), int(((p1[1] + p2[1]) / 2))
            path = [loc] + path
            a = frame
            # b = a[p1[1]:p2[1],p1[0]:p2[0]]
            x = 30
            b = a[p1[1] - x:p2[1] + x, p1[0] - x:p2[0] + x]
            worldmap[p1[1] - x:p2[1] + x, p1[0] - x:p2[0] + x,:] = b
            worldmap[findme] = 255
            # framearray[z] = worldmap # this is uncessary for final
            nav = gray2BG(navigable(b))
            edge= gray2red(findedges(nav))
            worldmap2[p1[1] - x:p2[1] + x, p1[0] - x:p2[0] + x,:] = (nav+edge)
            move = moveoptions(nav)
            findme = (200,200)

            next = nextmove(move,loc,findme)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            # cv2.line(frame,loc,next,(255,0,0),3)
            z = z+1 # this is uneccessary for final
            if len(path) >= 2:
                for i in range(0,len(path)-1):
                    cv2.line(frame,path[i],path[i+1],(0.255,0),2)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)
        cv2.imshow("Worldmap", worldmap/255)
        cv2.imshow("nav", worldmap2)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break