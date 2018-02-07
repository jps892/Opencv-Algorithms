from __future__ import division
import numpy as np
import cv2
import cv2.aruco as aruco
import glob

# define variable for resize tratio
ratio = 1

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def for_point_warp(cnt, orig):
    # we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # summing the (x, y) coordinates together by specifying axis=1
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Notice how our points are now stored in an imposed order:
    # top-left, top-right, bottom-right, and bottom-left.
    # Keeping a consistent order is important when we apply our perspective transformation

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    print rect, 'rect'
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    height, width, channels = orig.shape
    print height, width

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [rect[0][0]+2*width, rect[0][1]+4*height],
        [rect[0][0]+maxWidth - 1+2*width, rect[0][1]+4*height],
        [rect[0][0]+maxWidth - 1+2*width, rect[0][1]+maxWidth - 1+4*height],
        [rect[0][0]+2*width, rect[0][1]+maxWidth - 1+4*height]], dtype = "float32")

    print dst

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (4*width, 8*height))
    imgray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    contour = contours[0]
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if(area<cnt_area):
            area = cnt_area
            contour = cnt
    # warp = cv2.drawContours(warp, [contour], 0, (0, 255, 0), 3)

    recti = cv2.boundingRect(contour)
    print recti, "recti"
    crop = warp[recti[1]:(recti[1]+recti[3]), recti[0]:(recti[0]+recti[2])]

    # recti = cv2.minAreaRect(contour)
    # crop = crop_minAreaRect(warp, recti)


    return crop

def resize(img, width=None, height=None, interpolation = cv2.INTER_AREA):
    global ratio
    w, h, _ = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height/h
        width = int(w*ratio)
        print(width)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width/w
        height = int(h*ratio)
        print(height)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


# read the image
frame = cv2.imread('/home/jps/Downloads/images_2/IMG_2569.jpg')
# frame = rotate_bound(frame, -180)
oheight, owidth, ochannels = frame.shape
s_img = frame
l_img = np.zeros((oheight+40,oheight+40,3), np.uint8)
l_img[:,:] = (255,255,255)
pheight, pwidth, pchannels = l_img.shape
x_offset=y_offset=20
l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
cv2.imwrite('/home/jps/jjjj.jpg',l_img)


print oheight, owidth, ochannels, "iijovor", l_img.shape
frame = l_img

x_offset=y_offset=50
# l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
frame1 = resize(frame, height=1080)

# operations on the frame come here
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

cv2.imshow("width", frame1)

flat_object = frame1
flat_object_resized = flat_object.copy()

# make a copy
flat_object_resized_copy = flat_object_resized.copy()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

corners, ids, rejectedImgPoints = aruco.detectMarkers(gray1, aruco_dict, parameters=parameters)


font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)


if np.all(ids != None):
    aruco.drawDetectedMarkers(frame, corners, ids) #Draw A square around the markers
    print corners[0][0][0][0], "corners[0]"

    x_mar = corners[0][0][0][0] - corners[0][0][1][0]
    y_mar = corners[0][0][0][1] - corners[0][0][2][1]
    ax = x_mar / 7
    ay = y_mar / 7
    L = [[corners[0][0][0][0], corners[0][0][0][1]],
         [corners[0][0][1][0], corners[0][0][1][1]],
         [corners[0][0][2][0], corners[0][0][2][1]],
         [corners[0][0][3][0], corners[0][0][3][1]]]

    our_cnt = np.array(L).reshape(-1, 1, 2).astype(np.int32)
    print type(our_cnt), "our_cnt",our_cnt.shape

    # draw a contour
    # print our_cnt
    cv2.drawContours(flat_object_resized_copy, [our_cnt], -1, (0, 255, 0), 3)
    warped = for_point_warp(our_cnt / ratio, flat_object)
    # warped = resize(warped, height=oheight)
    # warped = rotate_bound(warped, 180)

    cv2.imshow("Original image", flat_object_resized)
    cv2.imshow("Marked ROI", flat_object_resized_copy)
    cv2.imshow("Warped ROI", warped)

    # cv2.imwrite("/home/jps/Downloads/images_2/stb_aruco_wrap_result/without_rotation/IMG_2569.jpg", warped)

    # Display the resulting frame
    cv2.imshow('frame',frame)
cv2.waitKey()

# When everything done, release the capture
cv2.destroyAllWindows()