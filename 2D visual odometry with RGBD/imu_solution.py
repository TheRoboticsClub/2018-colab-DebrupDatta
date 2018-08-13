from pyquaternion import Quaternion
import numpy as np
class Imu_odom:
    def __init__(self):
        self.orientation_t_prev = 1311878223.60
        self.accelerometer_t_prev = 1311878223.60
        self.ax_prev = 0
        self.ay_prev = 0
        self.R_prev = np.eye(3 , dtype = float)
        self.t =np.zeros((3,1) , dtype = float)

        '''ax_current =data.imu['x']
        ay_current =data.imu['y']
        imu_t_current = data.imu_t
        qz=data.orientation['qz']
        qw=data.orientation['qw']
        orientation_t_current = data.orientation_t
        #color image
        #color_image = data.color_img
        #depth image
        depth_image = data.depth_img'''

    def get_imu_estimate(self,ax, ay , accelerometer_t , qz , qw , orientation_t ):
        self.ax_current = ax
        self.ay_current = ay
        self.accelerometer_t_current = accelerometer_t
        self.orientation_t_current  = orientation_t
        self.qz = qz
        self.qw = qw

        d_t = self.accelerometer_t_current - self.accelerometer_t_prev
        V_avg_x = ((self.ax_prev + self.ax_current) *d_t)/2
        V_avg_y = ((self.ay_prev + self.ay_current)*d_t)/2
        t_current = np.zeros((3,1) , dtype = float)
        t_current[0,0] = V_avg_x * d_t
        t_current[1,0] = V_avg_y * d_t
        Q = Quaternion(self.qw , 0,0,self.qz)

        self.t = self.t + np.dot( self.R_prev , t_current)
        self.R_current = Q.rotation_matrix
        print(self.R_current)

        x = self.t[0,0] *1.61
        y = self.t[1,0] *1.61
        '''
        #Show processed image on GUI
        self.set_processed_image(depth_image)
        #set predicted pose
        self.set_predicted_pose(-y,x)'''
        self.accelerometer_t_prev = self.accelerometer_t_current
        self.orientation_t_prev = self.orientation_t_current
        self.ax_prev = self.ax_current
        self.ay_prev = self.ay_current
        self.R_prev = self.R_current

        return ( -y , x)