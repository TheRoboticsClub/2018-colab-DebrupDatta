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
class Mono_visual_odom:
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
        

    def estimate_motion(self):
    
        essential_mat,k1=cv2.findEssentialMat(self.keypoints_prev , self.keypoints_new ,focal = self.fx , pp = (self.cx , self.cy), method = cv2.RANSAC ,prob=0.999, threshold=1.0)
        _,R ,t ,k2 = cv2.recoverPose(essential_mat , self.keypoints_prev , self.keypoints_new, focal = self.fx , pp = (self.cx , self.cy))
        
        q = Quaternion(matrix = R)
        q[1] = 0
        q[3] = 0
        R = q.rotation_matrix
        

        t[1]= 0
        
        return (R ,t)
    def get_3D_pointcloud2(self, R, t):
        """Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""
        K = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]])
        # The canonical matrix (set as the origin)
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = K.dot(P0)
        # Rotated and translated using P0 as the reference point
        P1 = np.hstack((R, t))
        P1 = K.dot(P1)
        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
        point1 = self.keypoints_prev.reshape(2, -1)
        point2 = self.keypoints_new.reshape(2, -1)

        self.cloud_new = cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]


    def estimate_scale(self):
        if self.cloud_prev is None:
            self.cloud_prev = self.cloud_new
        min_idx = min([self.cloud_prev.shape[0], self.cloud_new.shape[0]])
        ratios = []  # List to obtain all the ratios of the distances
        '''for i in range(0,min_idx):
            if i > 0:
                Xk = cloud_new[i]
                p_Xk = cloud_new[i - 1]
                Xk_1 = cloud_prev[i]
                p_Xk_1 = cloud_prev[i - 1]

                if np.linalg.norm(p_Xk - Xk).all() != 0:
                    ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))'''

        for i in range(0,min_idx):
            for j in range(0, min_idx):
                if i != j and i%5==0 and j%5==1:
                    Xk = self.cloud_new[i]
                    p_Xk = self.cloud_new[j]
                    Xk_1 = self.cloud_prev[i]
                    p_Xk_1 = self.cloud_prev[j]

                    if np.linalg.norm(p_Xk - Xk).all() != 0:
                        ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

        d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
        return d_ratio

    def construct_trajectory(self, R,t,scale):
        '''
        pos_r_new = np.dot(R,pos_r)
        pos_t_new = pos_t + np.dot(pos_r_new,t) * scale
        if math.sqrt(((pos_t_new[0] - X[-1])**2)+((pos_t_new[2] - Y[-1])**2)) < 2 :
            pos_t = pos_t_new
            pos_r = pos_r_new
        else:
            pos_t=pos_t
            pos_r =pos_r
        '''
        q = Quaternion(matrix = R)
        if abs(q.degrees) < 5:
            #print('degrees_accept' ,q.degrees)
            pos_r_new = np.dot(R,self.pos_r)
        elif  177 <= abs(q.degrees) <=180:
            print('degrees_accept' ,q.degrees)
            print('t',t)
            #pos_r_new = self.pos_r
            #print(R)
            
            q_new = Quaternion(axis=[0,1,0], degrees=180-abs(q.degrees))
            R = q_new.rotation_matrix
            
            pos_r_new = np.dot(R,self.pos_r)
            #print('-----')
        else:
            print('degrees_reject' ,q.degrees)

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
        '''else:
            #print(t)
            #print("t",dist_diff)
            self.pos_r = pos_r_new
            return (self.pos_t[0] , self.pos_t[2] )'''


    def get_vo_estimate(self, color_image , depth_image):

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
            R,t=self.estimate_motion()
            self.get_3D_pointcloud2(R ,t)
            self.count = self.count +1
        else:
            self.detectfeatures()
            self.trackandshowfeatures()
            R,t=self.estimate_motion()
            self.get_3D_pointcloud2(R ,t)
            self.count=0


        scale = self.estimate_scale()
        print('scale' , scale)
        x,y= self.construct_trajectory(R , t , scale)
        #diff.append(math.sqrt(((x-X[-1])**2)+((y-Y[-1])**2)))
        #window = RT_trajectory_window ( x ,y ,window)
        #X.append(x)
        #Y.append(y)
        self.keypoints_prev  = self.keypoints_new
        self.img_prev_color = self.img_new_color        
        self.cloud_prev = self.cloud_new
        self.img_prev_gray = self.img_new_gray

        x = float(x[0]) / 35 #60 for monoVo 35 for kf with imu
        y = float(y[0]) / 35 #60
        
        return (-y ,-x , self.tracking_img)


    def get_rot_trans(self, color_image , depth_image):
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
            R,t=self.estimate_motion()
            self.get_3D_pointcloud2(R ,t)
            self.count = self.count +1
        else:
            self.detectfeatures()
            self.trackandshowfeatures()
            R,t=self.estimate_motion()
            self.get_3D_pointcloud2(R ,t)
            self.count=0


        scale = self.estimate_scale()
        
        q = Quaternion(matrix = R)
        if abs(q.degrees) < 5:
            R = R
        elif 177 <= abs(q.degrees) <=180:
            q_new = Quaternion(axis=[0,1,0], degrees=180-abs(q.degrees))
            R = q_new.rotation_matrix
        else:
            R = np.eye(3 ,dtype= float)

        if t[2]<0:
            t[2]=-t[2]
        
        self.keypoints_prev  = self.keypoints_new
        self.img_prev_color = self.img_new_color        
        self.cloud_prev = self.cloud_new
        self.img_prev_gray = self.img_new_gray
        return (R, t , scale ,self.tracking_img)