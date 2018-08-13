import os
import numpy as np
import cv2
import math
from pyquaternion import Quaternion
# FAST image detection parameters
fast_threshold_tight = 20
fast_threshold_loose = 10
#scale
scale=1
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (50,50), # 50,50
                  maxLevel = 30,#20 ,10 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) #10 ,0.03

class Rgbd_visual_odom():
    def __init__(self):
        self.fast = cv2.FastFeatureDetector_create()
        self.pos_t=np.zeros((3,1),dtype=float)
        self.pos_r=np.eye(3 , dtype =float)

        self.fx = 525  # focal length x
        self.fy = 525  # focal length y
        self.cx = 319.5  # optical center x
        self.cy = 239.5 # optical center y

        self.count = 0
        self.x=0
        self.keypoints_new = []
        self.estimated_coors=[]
        self.X=[]
        self.Y=[]
        self.Z=[]
        self.X.append(0)
        self.Y.append(0)
        self.diff=[]
        self.img_prev_color = None
        self.img_prev_no =None
        self.keypoints_prev = None
        self.cloud_prev =None
        self.img_prev_gray = None
        self.depth_img_prev = None
    def detectfeatures(self):
        up_segment = self.img_prev_color[0:240,:,:]
        down_segment = self.img_prev_color[240:,:,:]
        keypoints_prev1 = self.fast.detect(up_segment,None)
        keypoints_prev1 = np.array([k.pt for k in keypoints_prev1] ,dtype=np.float32)
        if len(keypoints_prev1) < 700 :
            
            self.fast.setThreshold(fast_threshold_loose)
            keypoints_prev1=self.fast.detect(up_segment)
            keypoints_prev1 = np.array([k.pt for k in keypoints_prev1] ,dtype=np.float32)
            self.fast.setThreshold(fast_threshold_tight)
        keypoints_prev2 = self.fast.detect(down_segment,None)
        keypoints_prev2 = np.array([k.pt for k in keypoints_prev2] ,dtype=np.float32)
        if len(keypoints_prev2) < 200 :
            
            self.fast.setThreshold(fast_threshold_loose)
            keypoints_prev2=self.fast.detect(down_segment)
            keypoints_prev2 = np.array([k.pt for k in keypoints_prev2] ,dtype=np.float32)
            self.fast.setThreshold(fast_threshold_tight)
        #print(keypoints_prev2)
        #print('-----------------')
        keypoints_prev2 = np.array([[k[0] , k[1]+240]  for k in keypoints_prev2] , dtype=np.float32)
        #print(keypoints_prev2)
        self.keypoints_prev = np.vstack((keypoints_prev1 ,keypoints_prev2))


    def trackandshowfeatures(self):
        self.img_new_gray = cv2.cvtColor(self.img_new_color , cv2.COLOR_BGR2GRAY)
        if self.img_prev_gray is None:
            self.img_prev_gray =self.img_new_gray
        #self.img_prev_gray = cv2.cvtColor(self.img_prev_color, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        
        self.keypoints_new, st, err = cv2.calcOpticalFlowPyrLK(self.img_prev_gray, self.img_new_gray, self.keypoints_prev, None, **lk_params)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.img_new_color)
        # draw the tracks
        remove=[]
        for i in range(0,len(self.keypoints_new)):
            if st[i]  == 1 and self.keypoints_new[i][0] <= 640 and self.keypoints_new[i][0] >= 0 and self.keypoints_new[i][1] <= 480 and self.keypoints_new[i][1] >= 0:
                
                a= self.keypoints_new[i][0]
                b= self.keypoints_new[i][1]
                c= self.keypoints_prev[i][0]
                d = self.keypoints_prev[i][1]
                
                color= np.random.randint(0,255,(3)).tolist()
                
                mask1  = cv2.circle(self.img_new_color,(a,b),4,color ,thickness = -1)
                mask2 = cv2.line(mask, (a,b),(c,d), color, 3 ,lineType = 8)
                #pass
            else:
                
                remove.append(i)
        self.keypoints_new = np.delete(self.keypoints_new ,remove, 0)
        self.keypoints_prev = np.delete(self.keypoints_prev, remove,0)

        self.tracking_img = cv2.add(mask1,mask2)


        '''cv2.imshow('frame',tracking_img)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            return''' 
    
    def construct_trajectory_rgbd(self, R,t):
        print("R",R)
        t= np.reshape(t , (3,1))
        print("t",t)
        q = Quaternion(matrix = R)
        if abs(q.degrees) < 5:
            #print('degrees_accept' ,q.degrees)
            pos_r_new = np.dot(R,self.pos_r)
        elif  177 <= abs(q.degrees) <=180:
            #print('degrees_accept' ,q.degrees)
            #pos_r_new = self.pos_r
            #print(R)
            q_new = Quaternion(axis=[0,1,0], degrees=180-abs(q.degrees))
            R = q_new.rotation_matrix
            pos_r_new = np.dot(R,self.pos_r)
            #print('-----')
        else:
            #print('degrees_reject' ,q.degrees)
            pos_r_new = self.pos_r
        #dist_diff = math.sqrt((np.dot(pos_r_new , t) * scale)[0]**2+ (np.dot(pos_r_new , t) * scale)[2]**2 )
        '''
        self.pos_r = pos_r_new
        self.pos_t = self.pos_t + np.dot(self.pos_r , t) * scale

        return (self.pos_t[0] , self.pos_t[2] )'''
        #if dist_diff < 2 :
        
        #if (t[2] > (t[0] -1.2) and t[2] > (t[1] - 0.1)) : #1.2 , 0.1
            #print(dist_diff)
        if t[2]<0:
            t[2]=-t[2]

        
        #scale = scale *1.08
        self.pos_r = pos_r_new
        self.pos_t = self.pos_t + np.dot(self.pos_r , t) * scale

        return (self.pos_t[0] , self.pos_t[2] )

    def get_3D_pointcloud_prev(self):
        self.cloud_prev=[]
        factor = 1 # for the 32-bit float images in the ROS bag files
        self.pc_prev_nan=[]
        for i , points in enumerate(self.keypoints_prev):
            if np.isnan(self.depth_img_prev[int(points[1]), int(points[0])]) != True:
                z = self.depth_img_prev[int(points[1]), int(points[0])] /factor
                u = points[0]
                v = points[1]
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy


                self.cloud_prev.append([x,y,z])
                
            else:
                self.cloud_prev.append([0,0,0])

        #self.cloud_prev = np.array(self.cloud_prev)

    def get_3D_pointcloud_new(self):

        self.cloud_new=[]
        
        #factor = 5000 # for the 16-bit PNG files
        factor = 1 # for the 32-bit float images in the ROS bag files
        self.pc_new_nan =[]
        for i, points in enumerate(self.keypoints_new):

            
            if (np.isnan(self.depth_img_new[int(points[1]), int(points[0])]) != True):

                z = self.depth_img_new[int(points[1]), int(points[0])] /factor
                u = points[0]
                v = points[1]
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy


                self.cloud_new.append([x,y,z])
                
            else:
                self.cloud_new.append([0,0,0])

        #self.cloud_new = np.array(self.cloud_new)
        

    def estimate_rgbd_motion(self):
        cloud_rem=[]
        scale = np.zeros((3,1) , dtype= float)
        for i in range(0, len(self.keypoints_new)):
            if self.cloud_new[i] == [0,0,0] or self.cloud_prev[i] == [0,0,0]:
                cloud_rem.append(i)


        self.cloud_prev = np.delete(self.cloud_prev , cloud_rem ,0)
        self.cloud_new = np.delete(self.cloud_new , cloud_rem , 0)

        
        
        retval, out, inliers = cv2.estimateAffine3D(src=self.cloud_prev,dst=self.cloud_new,ransacThreshold=3, confidence=0.99)
        
        print("out" , out)
        t= out[:,3]
        scale[0,0] = math.sqrt((out[0,0] **2)+ (out[1,0] **2) + (out[2,0] **2))
        scale[1,0] = math.sqrt((out[0,1] **2)+ (out[1,1] **2) + (out[2,1] **2))
        scale[2,0] = math.sqrt((out[0,2] **2)+ (out[1,2] **2) + (out[2,2] **2))
        if out[0,0] <0:
            scale[0,0] = -scale[0,0]
        if out[1,1] < 0 :
            scale[1,0] = -scale[1,0]
        if out[2,2] < 0:
            scale[2,0] = -scale[2,0]

        print("scale" ,scale)
        R = out[:,0:3]
        R[:,0] = R[:,0] / scale[0,0]
        R[:,1] = R[:,1] / scale[1,0]
        R[:,2] = R[:,2] / scale[2,0]
        print("Det R",np.linalg.det(R))
        return (R ,t)

    def get_rgbd_estimate(self, color_image , depth_image):
        self.depth_img_new= depth_image
        if self.depth_img_prev is None :
            self.depth_img_prev = depth_image
        self.img_new_color = color_image
        self.img_new_color = cv2.bilateralFilter(self.img_new_color , 9, 20,20)
        if self.x == 0:
            self.img_prev_color = color_image
            self.detectfeatures()
            self.x=1

        self.trackandshowfeatures()
        if len(self.keypoints_new) > 1000 and self.count < 10 :
            #keypoints_new ,keypoints_prev =trackandshowfeatures(img_prev_color , img_new_color , keypoints_prev)

            #depth_new = nearest_depth_file(img_new_no)
            self.get_3D_pointcloud_prev()
            self.get_3D_pointcloud_new()
            R,t=self.estimate_rgbd_motion()
            
            self.count = self.count +1
        else:
           
            self.detectfeatures()
            self.trackandshowfeatures()
            self.get_3D_pointcloud_prev()
            self.get_3D_pointcloud_new()
            R,t=self.estimate_rgbd_motion()
            
            self.count=0


        #scale = self.estimate_scale()
        x,y= self.construct_trajectory_rgbd(R , t)
        #diff.append(math.sqrt(((x-X[-1])**2)+((y-Y[-1])**2)))
        #window = RT_trajectory_window ( x ,y ,window)
        #X.append(x)
        #Y.append(y)
        self.depth_img_prev = self.depth_img_new
        self.keypoints_prev  = self.keypoints_new
        self.img_prev_color = self.img_new_color        
        self.cloud_prev = self.cloud_new
        self.img_prev_gray = self.img_new_gray

        #x= float(x[0]) /35
        #y= float(y[0]) /35
        
        return (x[0] , y[0] , self.tracking_img)
    
