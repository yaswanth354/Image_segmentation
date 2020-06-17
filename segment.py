##import numpy as np
##import cv2
##from matplotlib import pyplot as plt
##
##img = cv2.imread('second.jpeg')
##gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
##
##cv2.imshow('img',img)
##cv2.imshow("thre",thresh)
##print (ret,thresh)
##
##kernel = np.ones((3,3),np.uint8)
##opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
##
### sure background area
##sure_bg = cv2.dilate(opening,kernel,iterations=3)
##
### Finding sure foreground area
##dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
##ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
##
### Finding unknown region
##sure_fg = np.uint8(sure_fg)
##unknown = cv2.subtract(sure_bg,sure_fg)
##ret, markers = cv2.connectedComponents(sure_fg)
##
### Add one to all labels so that sure background is not 0, but 1
##markers = markers+1
##
### Now, mark the region of unknown with zero
##markers[unknown==255] = 0
##markers = cv2.watershed(img,markers)
##img[markers == -1] = [0,0,255]
##cv2.imshow('out',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

##
##
##
 #THRESHOLD SEGMENTATION


##
##from skimage.color import rgb2gray
##import numpy as np
##import cv2
##import matplotlib.pyplot as plt
##
##from scipy import ndimage
##
##
##image = cv2.imread('second.jpeg') 
##
##plt.imshow(image)
##cv2.imshow('in',image)
##
##gray = rgb2gray(image)
##plt.imshow(gray, cmap='gray')
##
##gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
##for i in range(gray_r.shape[0]):
##    if gray_r[i] > gray_r.mean():
##        gray_r[i] = 1
##    else:
##        gray_r[i] = 0
##gray = gray_r.reshape(gray.shape[0],gray.shape[1])
##plt.imshow(gray, cmap='gray')
##cv2.imshow('gr',gray)
##cv2.waitKey(0)
##cv2.destroyAllWindows()




###EDGE DETECTION
##

##import numpy as np
##import cv2
##from matplotlib import pyplot as plt
##from skimage.color import rgb2gray
##from scipy import ndimage
##
##image = plt.imread('second.jpeg')
##plt.imshow(image)
##
##
##
##gray = rgb2gray(image)
##
### defining the sobel filters
##sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
##print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
## 
##sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
##print(sobel_vertical, 'is a kernel for detecting vertical edges')
##
##
##
##out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
##out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
### here mo#de determines how the input array is extended when the filter overlaps a border
##
##plt.imshow(out_h, cmap='gray')
##kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
##print(kernel_laplace, 'is a laplacian kernel')
##
##out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
##plt.imshow(out_l, cmap='gray')
##cv2.imshow('outtt',out_l)
##
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##
###CLUSTERING
##
##
##import numpy as np
##import cv2
##from matplotlib import pyplot as plt
##from skimage.color import rgb2gray
##from scipy import ndimage
##
##
##
##pic = plt.imread('second.jpeg')/255  # dividing by 255 to bring the pixel values between 0 and 1
##print(pic.shape)
##plt.imshow(pic)
##
##
##pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
##pic_n.shape
##
##
##from sklearn.cluster import KMeans
##kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
##pic2show = kmeans.cluster_centers_[kmeans.labels_]
##
##
##cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
##plt.imshow(cluster_pic)
##
##
##cv2.imshow('output',cluster_pic)
##cv2.waitKey(0)
##cv2.destroyAllWindows()






##
##
##import math 
##
##from PIL import Image
##from pylab import *
##import matplotlib.cm as cm
##import scipy as sp
##import random
##
##im = Image.open('/home/sri/Pictures/pics/tigerin.jpg').convert('L')
##arr = np.asarray(im)
##
####
##
##rows,columns = np.shape(arr)
###print '\nrows',rows,'columns',columns
##plt.figure()
##plt.imshow(im)
##plt.gray()
###User selects the intial seed point
##print ('\nPlease select the initial seed point')
##
##pseed = plt.ginput(1)
###pseed
###print pseed[0][0],pseed[0][1]
##
##x = int(pseed[0][0])
##y = int(pseed[0][1])
##x = int(179)
##y = int(86)
##seed_pixel = []
##seed_pixel.append(x)
##seed_pixel.append(y)
##
##print ('you clicked:',seed_pixel)
##
###closing figure
##plt.close()
##
##img_rg = np.zeros((rows+1,columns+1))
##img_rg[seed_pixel[0]][seed_pixel[1]] = 255.0
##img_display = np.zeros((rows,columns))
##
##region_points = []
##region_points.append([x,y])
##
##def find_region():
##	print ('\nloop runs till region growing is complete')
##	#print ('starting points',i,j)
##	count = 0
##	x = [-1, 0, 1, -1, 1, -1, 0, 1]
##	y = [-1, -1, -1, 0, 0, 1, 1, 1]
##	
##	while( len(region_points)>0):
##		
##		if count == 0:
##			point = region_points.pop(0)
##			i = point[0]
##			j = point[1]
##		print ('\nloop runs till length become zero:')
##		print ('len',len(region_points))
##		#print ('count',count )
##		val = arr[i][j]
##		lt = val - 8
##		ht = val + 8
##		#print ('value of pixel',val)
##		for k in range(8):	
##			#print ('\ncomparison val:',val, 'ht',ht,'lt',lt)
##		    if img_rg[i+x[k]][j+y[k]] !=1:
##			    try:
##				    if  arr[i+x[k]][j+y[k]] > lt and arr[i+x[k]][j+y[k]] < ht:
##					    #print '\nbelongs to region',arr[i+x[k]][j+y[k]]
##					    img_rg[i+x[k]][j+y[k]]=1
##					    p = [0,0]
##					    p[0] = i+x[k]
##					    p[1] = j+y[k]
##					    if p not in region_points: 
##					        if 0< p[0] < rows and 0< p[1] < columns:
##								
##						        region_points.append([i+x[k],j+y[k]])
##				    else:
##					    #print 'not part of region'
##					    img_rg[i+x[k]][j+y[k]]=0
##			    except IndexError:     
##                			        continue
##
##		#print '\npoints list',region_points
##		point = region_points.pop(0)
##		i = point[0]
##		j = point[1]
##		count = count +1
##		#find_region(point[0], point[1])			 
##		
##find_region()
##
##ground_out = np.zeros((rows,columns))
##
##for i in range(rows):
##	for j in range(columns):
##		if arr_out[i][j] >125:
##			ground_out[i][j] = int(1)
##
##		else:
##			ground_out[i][j] = int(0)
##
##
##tp = 0
##tn = 0
##fn = 0
##fp = 0
##
##for i in range(rows):
##	for j in range(columns):
##		if ground_out[i][j] == 1 and img_rg[i][j] == 1:
##			tp = tp + 1
##		if ground_out[i][j] == 0 and img_rg[i][j] == 0:
##			tn = tn + 1
##		if ground_out[i][j] == 1 and img_rg[i][j] == 0:
##			fn = fn + 1
##		if ground_out[i][j] == 0 and img_rg[i][j] == 1:
##			fp = fp + 1
##''' ********************************** Calculation of Tpr, Fpr, F-Score ***************************************************'''
##
##print ('\n************Calculation of Tpr, Fpr, F-Score********************')
##
###TP rate = TP/TP+FN
##tpr= float(tp)/(tp+fn)
##print ("\nTPR is:",tpr)
##
###fp rate is
##fpr= float(fp)/(fp+tn)
##print ("\nFPR is:",fpr)
##
###F-score as 2TP/(2TP + FP + FN)
##fscore = float(2*tp)/((2*tp)+fp+fn)
##print ("\nFscore:",fscore)
##
##
##plt.figure()
##plt.imshow(img_rg, cmap="Greys_r")
##plt.colorbar()
##plt.show()








##
##import cv2
##import numpy as np
##import random
##import sys
##
###class pour une pile
##class Stack():
##    def __init__(self):
##        self.item = []
##        self.obj=[]
##    def push(self, value):
##        self.item.append(value)
##
##    def pop(self):
##        return self.item.pop()
##
##    def size(self):
##        return len(self.item)
##
##    def isEmpty(self):
##        return self.size() == 0
##
##    def clear(self):
##        self.item = []
##
##class regionGrow():
##  
##    def __init__(self,im_path,th):
##        self.readImage(im_path)
##        self.h, self.w,_ =  self.im.shape
##        self.passedBy = np.zeros((self.h,self.w), np.double)
##        self.currentRegion = 0
##        self.iterations=0
##        self.SEGS=np.zeros((self.h,self.w,3), dtype='uint8')
##        self.stack = Stack()
##        self.thresh=float(th)
##    def readImage(self, img_path):
##        self.im = cv2.imread('image1.jpg',1)
##    
##
##    def getNeighbour(self, x0, y0):
##        neighbour = []
##        for i in (-1,0,1):
##            for j in (-1,0,1):
##                if (i,j) == (0,0): 
##                    continue
##                x = x0+i
##                y = y0+j
##                if self.limit(x,y):
##                    neighbour.append((x,y))
##        return neighbour
##    def ApplyRegionGrow(self):
##        randomseeds=[[self.h/2,self.w/2],
##            [self.h/3,self.w/3],[2*self.h/3,self.w/3],[self.h/3-10,self.w/3],
##            [self.h/3,2*self.w/3],[2*self.h/3,2*self.w/3],[self.h/3-10,2*self.w/3],
##            [self.h/3,self.w-10],[2*self.h/3,self.w-10],[self.h/3-10,self.w-10]
##                    ]
##        np.random.shuffle(randomseeds)
##        for x0 in range (self.h):
##            for y0 in range (self.w):
##         
##                if self.passedBy[x0,y0] == 0 and (int(self.im[x0,y0,0])*int(self.im[x0,y0,1])*int(self.im[x0,y0,2]) > 0) :  
##                    self.currentRegion += 1
##                    self.passedBy[x0,y0] = self.currentRegion
##                    self.stack.push((x0,y0))
##                    self.prev_region_count=0
##                    while not self.stack.isEmpty():
##                        x,y = self.stack.pop()
##                        self.BFS(x,y)
##                        self.iterations+=1
##                    if(self.PassedAll()):
##                        break
##                    if(self.prev_region_count<8*8):     
##                        self.passedBy[self.passedBy==self.currentRegion]=0
##                        x0=random.randint(x0-4,x0+4)
##                        y0=random.randint(y0-4,y0+4)
##                        x0=max(0,x0)
##                        y0=max(0,y0)
##                        x0=min(x0,self.h-1)
##                        y0=min(y0,self.w-1)
##                        self.currentRegion-=1
##
##        for i in range(0,self.h):
##            for j in range (0,self.w):
##                val = self.passedBy[i][j]
##                if(val==0):
##                    self.SEGS[i][j]=255,255,255
##                else:
##                    self.SEGS[i][j]=val*35,val*90,val*30
##        if(self.iterations>200000):
##            print("Max Iterations")
##        print("Iterations : "+str(self.iterations))
##        cv2.imshow("",self.SEGS)
##        cv2.waitKey(0)
##        cv2.destroyAllWindows()
##    def BFS(self, x0,y0):
##        regionNum = self.passedBy[x0,y0]
##        elems=[]
##        elems.append((int(self.im[x0,y0,0])+int(self.im[x0,y0,1])+int(self.im[x0,y0,2]))/3)
##        var=self.thresh
##        neighbours=self.getNeighbour(x0,y0)
##        
##        for x,y in neighbours:
##            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0)<var:
##                if(self.PassedAll()):
##                    break;
##                self.passedBy[x,y] = regionNum
##                self.stack.push((x,y))
##                elems.append((int(self.im[x,y,0])+int(self.im[x,y,1])+int(self.im[x,y,2]))/3)
##                var=np.var(elems)
##                self.prev_region_count+=1
##            var=max(var,self.thresh)
##                
##    
##    
##    def PassedAll(self):
##   
##        return self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.w*self.h
##
##
##    def limit(self, x,y):
##        return  0<=x<self.h and 0<=y<self.w
##    def distance(self,x,y,x0,y0):
##        return ((int(self.im[x,y,0])-int(self.im[x0,y0,0]))**2+(int(self.im[x,y,1])-int(self.im[x0,y0,1]))**2+(int(self.im[x,y,2])-int(self.im[x0,y0,2]))**2)**0.5
##
##
##
##
##exemple = regionGrow(sys.argv[0],'13')
##exemple.ApplyRegionGrow()











#region based
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage


##
##image = plt.imread("/home/sri/Pictures/pics/main1.jpg")
##image.shape
##plt.imshow(image)
##gray = rgb2gray(image)
##plt.imshow(gray, cmap='gray')
##
##
##
##gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
##for i in range(gray_r.shape[0]):
##    if gray_r[i] > gray_r.mean():
##        gray_r[i] = 1
##    else:
##        gray_r[i] = 0
##gray = gray_r.reshape(gray.shape[0],gray.shape[1])
##plt.imshow(gray, cmap='gray')
##
##
##gray = rgb2gray(image)
##gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
##for i in range(gray_r.shape[0]):
##    if gray_r[i] > gray_r.mean():
##        gray_r[i] = 3
##    elif gray_r[i] > 0.5:
##        gray_r[i] = 2
##    elif gray_r[i] > 0.25:
##        gray_r[i] = 1
##    else:
##        gray_r[i] = 0
##gray = gray_r.reshape(gray.shape[0],gray.shape[1])
##plt.imshow(gray, cmap='gray')
##









##
##import numpy as np
##
##import cv2
##im=cv2.imread('image1.jpg')
##cv2.imshow('in',im)
##image = cv2.imread("image1.jpg",0)
##output = cv2.imread("image1.jpg",1)
##cv2.imshow("Original image", image)
##cv2.waitKey()
##
##blurred = cv2.GaussianBlur(image,(11,11),0)
##
##cv2.imshow("Blurred image", blurred)
##cv2.waitKey()
##
##circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
##                             param1=100,param2=90,minRadius=0,maxRadius=200)
##
### cv2.HoughCircles function has a lot of parameters, so you can find more about it in documentation
### or you can use cv2.HoughCircles? in jupyter nootebook to get that 
##
### Check to see if there is any detection
##if circles is not None:
##    # If there are some detections, convert radius and x,y(center) coordinates to integer
##    circles = np.round(circles[0, :]).astype("int")
##
##    for (x, y, r) in circles:
##        # Draw the circle in the output image
##        cv2.circle(output, (x, y), r, (0,255,0), 3)
##        # Draw a rectangle(center) in the output image
##        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0,255,0), -1)
##
##cv2.imshow("Detections",output)
##cv2.imwrite("CirclesDetection.jpg",output)
##cv2.waitKey(0)


import sys
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file = 'second.jpeg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
    gray = cv.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    
    
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])




##
##from scipy.spatial import distance as dist
##from imutils import perspective
##from imutils import contours
##import numpy as np
##import argparse
##import imutils
##import cv2
##def midpoint(ptA, ptB):
##	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
##
##image = cv2.imread('secedge.png')
##gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##gray = cv2.GaussianBlur(gray, (7, 7), 0)
### perform edge detection, then perform a dilation + erosion to
### close gaps in between object edges
##edged = cv2.Canny(gray, 50, 100)
##edged = cv2.dilate(edged, None, iterations=1)
##edged = cv2.erode(edged, None, iterations=1)
### find contours in the edge map
##cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
##	cv2.CHAIN_APPROX_SIMPLE)
##cnts = imutils.grab_contours(cnts)
### sort the contours from left-to-right and initialize the
### 'pixels per metric' calibration variable
##(cnts, _) = contours.sort_contours(cnts)
##pixelsPerMetric = None
##
##for c in cnts:
##	# if the contour is not sufficiently large, ignore it
##	if cv2.contourArea(c) < 100:
##		continue
##	# compute the rotated bounding box of the contour
##	orig = image.copy()
##	box = cv2.minAreaRect(c)
##	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
##	box = np.array(box, dtype="int")
##	# order the points in the contour such that they appear
##	# in top-left, top-right, bottom-right, and bottom-left
##	# order, then draw the outline of the rotated bounding
##	# box
##	box = perspective.order_points(box)
##	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
##	# loop over the original points and draw them
##	for (x, y) in box:
##		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
##	(tl, tr, br, bl) = box
##	(tltrX, tltrY) = midpoint(tl, tr)
##	(blbrX, blbrY) = midpoint(bl, br)
##	# compute the midpoint between the top-left and top-right points,
##	# followed by the midpoint between the top-righ and bottom-right
##	(tlblX, tlblY) = midpoint(tl, bl)
##	(trbrX, trbrY) = midpoint(tr, br)
##	# draw the midpoints on the image
##	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
##	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
##	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
##	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
##	# draw lines between the midpoints
##	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
##		(255, 0, 255), 2)
##	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
##		(255, 0, 255), 2)
##	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
##	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
##	# if the pixels per metric has not been initialized, then
##	# compute it as the ratio of pixels to supplied metric
##	# (in this case, inches)
##	if pixelsPerMetric is None:
##		pixelsPerMetric = dB / 12
##	dimA = dA / pixelsPerMetric
##	dimB = dB / pixelsPerMetric
##	# draw the object sizes on the image
##	cv2.putText(orig, "{:.1f}in".format(dimA),
##		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
##		0.65, (255, 255, 255), 2)
##	cv2.putText(orig, "{:.1f}in".format(dimB),
##		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
##		0.65, (255, 255, 255), 2)
##	# show the output image
##	cv2.imshow("Image", orig)
##	cv2.waitKey(0)
##	cv2.destroyAllWindows()
####	











##
##import cv2  
##import numpy as np  
## 
##image1 = cv2.imread('second.jpeg')  
##  
### cv2.cvtColor is applied over the 
### image input with applied parameters 
### to convert the image in grayscale  
##img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
##  
### applying different thresholding  
### techniques on the input image 
### all pixels value above 120 will  
### be set to 255 
##ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
##ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
### with the corresponding thresholding  
### techniques applied to the input images 
##cv2.imshow('Binary Threshold', thresh1) 
##cv2.imshow('Binary Threshold Inverted', thresh2) 
### De-allocate any associated memory usage   
##cv2.waitKey(0)
##cv2.destroyAllWindows()  
