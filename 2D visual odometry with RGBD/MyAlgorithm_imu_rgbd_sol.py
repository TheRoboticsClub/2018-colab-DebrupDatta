import threading
import time
import math
import rosbag
import cv2
import numpy as np
from datetime import datetime

#from visual_solution import Mono_visual_odom
from visual_sol_rgbd import Rgbd_visual_odom
from imu_solution import Imu_odom
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from pyquaternion import Quaternion
sigmas = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=1)#alpha =0.8

time_cycle = 40 #80

class MyAlgorithm(threading.Thread):

    def __init__(self, bag_readings, pose_obj):
        self.bag_readings = bag_readings
        self.pose_obj = pose_obj
        self.threshold_image = np.zeros((640,480,3), np.uint8)
        self.color_image = np.zeros((640,480,3), np.uint8)
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
        self.diff_time = 0

        #self.vo = Mono_visual_odom()
        self.imu_od = Imu_odom()
        self.rgbd_od = Rgbd_visual_odom()
        self.ukf = None
        self.accelerometer_t_prev = 1311878223.60
        self.ax_prev = 0
        self.ay_prev = 0
        self.R_prev = np.eye(3 , dtype = float)


    def getReadings(self , *sensors):
        self.lock.acquire()
        data = self.bag_readings.getData(sensors)
        self.lock.release()
        return data

    def set_predicted_path(self,path):
        self.pose_lock.acquire()
        
        self.pose_obj.set_pred_path(path)
        self.pose_lock.release()

    def set_predicted_pose(self,x,y):
        self.pose_lock.acquire()
        self.predicted_pose = [x,y]
        self.pose_obj.set_pred_pose([x,y])
        self.pose_lock.release()

    def get_predicted_pose(self):
        self.pose_lock.acquire()
        
    def set_processed_image(self,image):
        img = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.threshold_image_lock.acquire()
        self.threshold_image = img
        self.threshold_image_lock.release()
    def get_processed_image (self):
        self.threshold_image_lock.acquire()
        img  = np.copy(self.threshold_image)
        self.threshold_image_lock.release()
        return img

    def run (self):

        #self.algo_start_time = time.time()
        while (not self.kill_event.is_set()):
            start_time = datetime.now()

            if not self.stop_event.is_set():
                self.algo_start_time = time.time()
                self.algorithm()
                self.algo_stop_time = time.time()
                self.diff_time = self.diff_time + (self.algo_stop_time - self.algo_start_time)
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):
        #Getting readings data
        data = self.getReadings('accelerometer' , 'orientation' , 'color_img' , 'depth_img') # to get readings data from particular sensors
        '''
        data = self.getReadings() # to get data from all sensors
        data = self.getReadings('stream') # to stream data from all sensors one by one 
        '''
        
        #imu data

        ax=data.accelerometer['x']
        ay=data.accelerometer['y']
        accelerometer_t = data.accelerometer_t
        orientation_t = data.orientation_t
        qz=data.orientation['qz']
        qw=data.orientation['qw']
        #color image
        color_image = data.color_img
        #depth image
        depth_image = data.depth_img
        #x , y  = self.imu_od.get_imu_estimate( ax, ay , accelerometer_t , qz , qw , orientation_t)
        self.ax_current = ax
        self.ay_current = ay
        self.accelerometer_t_current = accelerometer_t
        self.orientation_t_current  = orientation_t
        self.qz = qz
        self.qw = qw

        self.d_t = self.accelerometer_t_current - self.accelerometer_t_prev


        def f_cv (x ,dt , ax_prev , ax_current  , ay_prev , ay_current , R_current):
            V_avg_x = ((ax_prev + ax_current) *dt)/2
            V_avg_y = ((ay_prev + ay_current)*dt)/2

            t_current = np.zeros((3,1) , dtype = float)
            t_current[0,0] = V_avg_x * dt
            t_current[1,0] = V_avg_y * dt
            x = np.reshape(x , (2,1))
            x = x + np.dot( R_current , t_current)[0:2,:]
            x = np.reshape(x ,2)
            #x[0] = x[0] *1.61
            #x[1] = x[1] *1.2

            
            return (x)


        

        def h_cv(x):
            print(x)
            return x


        if self.ukf == None:
            dt=1
            self.ukf = UKF(dim_x=2, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
            self.ukf.x = np.zeros((2) , dtype =float)
            self.ukf.P = np.array([[0.000001 , 0],
                                    [0,0.000001]] ,dtype= float)
            self.ukf.Q  = np.diag([(0.225**2),(0.225**2)]) #imu variance 
            self.ukf.R = np.diag([(0.005**2),(0.005**2)]) #rgbd variance 


        #imu_x , imu_y  = self.imu_od.get_imu_estimate( ax, ay , accelerometer_t , qz , qw , orientation_t)
        Q = Quaternion(self.qw , 0,0,self.qz)
        self.R_current = Q.rotation_matrix
        self.ukf.predict(dt = self.d_t , ax_prev = self.ax_prev , ax_current = self.ax_current  , ay_prev = self.ay_prev , ay_current = self.ay_current, R_current = self.R_current)
        print("imu", self.ukf.x)
        '''
        Q = Quaternion(self.qw , 0,0,self.qz)
        self.R_current = Q.rotation_matrix
        '''
        z = np.zeros((2) , dtype  = float)
        #z[0] ,z[1] , mask = self.vo.get_vo_estimate(color_image , depth_image)
        z[0] ,z[1] , mask = self.rgbd_od.get_rgbd_estimate(color_image , depth_image)
        print("vo", z)
        self.ukf.update(z)



        '''
        x ,y , mask = self.vo.get_vo_estimate(color_image , depth_image)
        x= float(x[0]) /35
        y= float(y[0]) /35
        print(x)
        print(y)
        #x = 1 
        #y = 1
        #Show processed image on GUI
        self.set_processed_image(mask)'''
        self.set_processed_image(mask)
        #print(self.ukf.P)
        #set predicted pose
        print("ukf" , self.ukf.x)
        self.set_predicted_pose(self.ukf.x[0],self.ukf.x[1])
        self.accelerometer_t_prev = self.accelerometer_t_current
        self.orientation_t_prev = self.orientation_t_current
        self.ax_prev = self.ax_current
        self.ay_prev = self.ay_current
        self.R_prev = self.R_current

        #set predicted path at once /or reset the previously set predicted poses at once ---- path should be Nx2 numpy array or python list [x,y].
        #self.set_predicted_path(path)