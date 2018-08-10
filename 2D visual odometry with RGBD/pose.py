import numpy as np
class Pose:
    def __init__(self):
        self.actual_pose = None
        self.pred_pose = None
        self.actual_path = []
        self.pred_path = []

    def set_pred_pose(self, pos):
        self.pred_pose = pos
        self.pred_path.append(pos)

    def set_pred_path(self,path):
        self.pred_path = path

    def get_pred_pose(self):
        return (self.pred_pose)

    def set_actual_pose(self,pos):
        if len(self.actual_path) == 0:
            self.init_actual_pose = pos
            self.actual_pose = [0,0]
            self.actual_path.append([0,0])
        self.actual_pose = [pos[0] - self.init_actual_pose[0] , pos[1] - self.init_actual_pose[1] ]
        self.actual_path.append(self.actual_pose)

    def get_actual_pose(self):
        return (self.actual_pose)

    def get_actual_path(self):
        np_actual_path = np.array(self.actual_path)
        return (np_actual_path)

    def get_pred_path(self):
        np_pred_path = np.array(self.pred_path)
        return (np_pred_path)