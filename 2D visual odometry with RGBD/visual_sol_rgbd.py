import os
import numpy as np
import time
import cv2
import math
from pyquaternion import Quaternion
#from networkx.algorithms import approximation
#import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import scipy.interpolate as interp

# FAST image detection parameters
fast_threshold_tight = 30 #20
fast_threshold_loose = 20 #10
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
        self.path = np.zeros((1,2)  ,dtype = float)
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
    def interpolate_polyline(self,polyline, num_points):
        duplicates = []
        for i in range(1, len(polyline)):
            if np.allclose(polyline[i], polyline[i-1]):
                duplicates.append(i)
        if duplicates:
            polyline = np.delete(polyline, duplicates, axis=0)
        if (len(polyline.T[0]) == 1):
            return polyline
        elif (len(polyline.T[0]) <= 3):
            k = len(polyline.T[0]) -1
        else:
            print("-------------------------")
            k=3 
        tck, u = interp.splprep(polyline.T, k=k ,s=0)
        u = np.linspace(0.0, 1.0, num_points)
        return np.column_stack(interp.splev(u, tck))

    def construct_trajectory_rgbd(self, R,t):
        print("R",R)
        #t= np.reshape(t , (3,1))
        print("t",t)
        q = Quaternion(matrix = R)
        print("degrees" , q.degrees)
        
        

        #pos_r_new = np.dot(R,self.pos_r)
        #scale = scale *1.08
        world_degree = Quaternion(matrix = self.pos_r).degrees
        
        self.pos_r = np.dot(R,self.pos_r)
        self.pos_t = self.pos_t + np.dot(self.pos_r, t)
        

        #self.pos_r = np.dot(R,self.pos_r)
        self.path = np.append(self.path , [[float(self.pos_t[0])  , float(self.pos_t[2])]] , axis =0)
        self.path = self.interpolate_polyline(self.path, len(self.path))
        x = self.path[-1][0]
        y = self.path[-1][1]
        return (x ,y)

    def get_3D_pointcloud_prev(self):
        self.cloud_prev=[]
        factor = 1 # for the 32-bit float images in the ROS bag files
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
        
    def match_cloud_shapes(self):
        cloud_rem=[]
        scale = np.zeros((3,1) , dtype= float)
        for i in range(0, len(self.keypoints_new)):
            if self.cloud_new[i] == [0,0,0] or self.cloud_prev[i] == [0,0,0]:
                cloud_rem.append(i)


        self.cloud_prev = np.delete(self.cloud_prev , cloud_rem ,0)
        self.cloud_new = np.delete(self.cloud_new , cloud_rem , 0)
        #print("cloud shape", self.cloud_prev.shape)
    
    def estimate_rgbd_motion(self):
        if self.inliers != []:
            
            self.cloud_prev_inliers = self.cloud_prev[self.inliers]
            self.cloud_new_inliers  = self.cloud_new[self.inliers]
        else:
            #print("[]")
            self.cloud_prev_inliers = self.cloud_prev
            self.cloud_new_inliers  = self.cloud_new


        
        cloud_prev_mean = self.cloud_prev_inliers.mean(0)
        cloud_new_mean = self.cloud_new_inliers.mean(0)
        cloud_prev_centered = np.subtract(self.cloud_prev_inliers , cloud_prev_mean)
        cloud_new_centered = np.subtract(self.cloud_new_inliers  ,cloud_new_mean)
    
        print("self.cloud_prev_inliers.shape[0]" , self.cloud_prev_inliers.shape[0])
        w = np.eye(self.cloud_prev_inliers.shape[0])
        s = np.dot(cloud_prev_centered.T , np.dot(w,cloud_new_centered))
        U_svd,D_svd,V_svd = np.linalg.linalg.svd(s)
        D_svd = np.diag(D_svd)
        V_svd = np.transpose(V_svd)
        z = np.eye(3)
        z[2,2]  = np.linalg.det(np.dot(V_svd, U_svd.T))
        R = np.dot(V_svd, np.dot(z, np.transpose(U_svd)))
        t = cloud_new_mean - np.dot(R , cloud_prev_mean)
        #R = R.T
        t = np.reshape(t, (3,1))
        
        q = Quaternion(matrix = R)
        q[1] = 0
        q[3] = 0
        R = q.rotation_matrix
        

        t[1,0]= 0
        #t[0,0] = abs(t[0,0])

        if t[2,0] > 0:
            t[2,0] = 0
            #t[0,0] = 0
            #R = np.eye(3)
            print("drop1")
        if abs(t[0,0]) > 0.1: 
            t[0,0] = 0
            R = np.eye(3)

        if abs(t[2,0]) > 0.1 :
            t[2,0] = 0
            #R = np.eye(3)  # with # best 



        #t[2,0] = -abs(t[2,0])
        #t[0,0] = -abs(t[0,0])        
        return (R ,t)

    def select_inliners(self):
        inliers = []

        dist_prev = euclidean_distances(self.cloud_prev , self.cloud_prev)
        dist_new = euclidean_distances(self.cloud_new , self.cloud_new)
        '''
        dist_prev = np.zeros((self.cloud_prev.shape[0],self.cloud_prev.shape[0]) , dtype=float)
        for i in range(0,self.cloud_prev.shape[0]):
            j = 0
            while j < i :
                dist_prev[i,j] = math.sqrt((self.cloud_prev[i][0] - self.cloud_prev[j][0])**2 + (self.cloud_prev[i][1] - self.cloud_prev[j][1])**2 + (self.cloud_prev[i][2] - self.cloud_prev[j][2])**2)
                j = j+1
        dist_new = np.zeros((self.cloud_new.shape[0],self.cloud_new.shape[0]) , dtype=float)
        for i in range(0,self.cloud_new.shape[0]):
            j = 0
            while j < i :
                dist_new[i,j] = math.sqrt((self.cloud_new[i][0] - self.cloud_new[j][0])**2 + (self.cloud_new[i][1] - self.cloud_new[j][1])**2 + (self.cloud_new[i][2] - self.cloud_new[j][2])**2)
                j = j + 1
        '''
        M = dist_new - dist_prev

        threshold = 0.00010 #0.00010
        consistency_mask = (M > -threshold) & (M < threshold)
        consistency_mask = consistency_mask * M
        #print("np.count_nonzero(consistency_mask)",np.count_nonzero(consistency_mask))
        

        nonzero_indices = np.nonzero(consistency_mask)
        inliers = np.intersect1d(nonzero_indices[0] , nonzero_indices[1])
        inliers  = np.unique(inliers)
        #inliers = nonzero_indices[0]

        '''
        for i in range (0,self.cloud_prev.shape[0]):
            j=0
            flag = 0
            while j<i:
                if abs(consistency_mask[i][j]) != 0:
                    flag = 1
                    break
                j = j+1
            if flag  ==1:
                if i not in inliers:
                    inliers.append(i)
                if j not in inliers:
                    inliers.append(j)
        '''

        '''
        g = nx.Graph()
        g.add_nodes_from(range(0, self.cloud_prev.shape[0]))
        for i in range (0,self.cloud_prev.shape[0]):
            j=0
            while j<i:
                if abs(consistency_mask[i][j]) != 0:
                    g.add_edge(i,j)
                j = j+1

        g.remove_nodes_from(list(nx.isolates(g)))
        maxclique = approximation.clique.max_clique(g)
        clique = clique + list(maxclique)
        count = 0
        '''
        '''while len(clique) <= 8 and count <=10:
            g.remove_nodes_from(list(maxclique))
            maxclique = approximation.clique.max_clique(g)
            clique = clique + list(maxclique)
            print("clique",clique)
            count=count+1s
        '''
        
        self.inliers = inliers


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
            self.match_cloud_shapes()
            t1 = time.time()
            self.select_inliners()
            print("select_inliners_time", time.time() - t1)
            R,t=self.estimate_rgbd_motion()
            
            self.count = self.count +1
        else:
           
            self.detectfeatures()
            self.trackandshowfeatures()
            self.get_3D_pointcloud_prev()
            self.get_3D_pointcloud_new()
            self.match_cloud_shapes()
            t1 = time.time()
            self.select_inliners()
            print("select_inliners_time", time.time() - t1)
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
        #time.sleep(3)
        return (y , x , self.tracking_img)
    
