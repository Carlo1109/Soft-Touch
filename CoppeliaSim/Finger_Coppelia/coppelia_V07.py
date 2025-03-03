from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from enum import Enum
from math import pi, cos, sin, sqrt 
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class SceneException(Exception):
    pass

class CoppeliaConnector:
    l1 = 0.51
    l2 = 0.31
    l3 = 0.23
    offset = 0.396
    p_ref = 0.639

    def __init__(self):
        self._client = RemoteAPIClient()
        self.sim = self._client.require('sim')

        self._check_correct_engine()
        
        self._fetch()
        
        self._initialization()

        self.force_calculator = SensorForceCalculator(self)

    def _check_correct_engine(self):
        engine = self.sim.getInt32Param(self.sim.intparam_dynamic_engine)
        if engine != self.sim.physics_mujoco:
            raise SceneException("The scene must run using the MuJoCo physics engine.")

    def _fetch(self):
        self.J1 = self.sim.getObject("/Base/J1")
        self.J2 = self.sim.getObject("/Base/J2")
        self.J3 = self.sim.getObject("/Base/J3")
        self.Tip = self.sim.getObject("/Base/FingerTip")
        self.L3 = self.sim.getObject("/Base/l3")


    def _initialization(self):
        self.graphHandle = self.sim.getObject("/Graph")
        self.graphD = self.sim.addGraphStream(self.graphHandle,'Distance','',0)
        self.graph_ref = self.sim.addGraphStream(self.graphHandle,'Reference','',0,[0.5,1,0]) #last vector is for color
        self.end_trace = self.sim.addDrawingObject(self.sim.drawing_linestrip,5,0,-1,100000,[1,0,0])

    def ini(self):
        self.sim.setJointPosition(self.J1, pi/8)
        self.sim.setJointPosition(self.J2, pi/8)
        self.sim.setJointPosition(self.J3, pi/8)
        pos = self.sim.getObjectPosition(self.Tip)
        x = pos[0]
        y = pos[2]
        
        self.filt = self.init_filter(x,y)
        
    def init_filter(self,x,y):
        filt = KalmanFilter(dim_x=4,dim_z=2)
        dt = 0.025
        filt.F = np.array(
            [[1,dt,0,0],
             [0,1,0,0],
             [0,0,1,dt],
             [0,0,0,1]]
        )
        filt.u = 0
        filt.H = np.array([[1,0,0,0],
                           [0,0,1,0]])
        filt.R = np.array([[0.2, 0],
                            [0, 0.2]])

        q = Q_discrete_white_noise(dim=2,dt=dt,var=0.5)
        filt.Q = block_diag(q,q)
        filt.x = np.array([[0,0,0,0]]).T
        filt.P = np.eye(4) * 2000
        return filt

    def sensing(self):
        end_pos = self.sim.getObjectPosition(self.Tip, self.sim.handle_world)
        self.sim.addDrawingObjectItem(self.end_trace, end_pos)
    
    def move_index(self, distance, d_pre):
        t = self.sim.getSimulationTime() #current time
        timeStep = self.sim.getSimulationTimeStep()

        # Current Position Joint Variables
        q1 = self.sim.getJointPosition(self.J1)
        q2 = self.sim.getJointPosition(self.J2)
        q3 = self.sim.getJointPosition(self.J3)

        # Cartesian task [z x] wrt World RF
        z = self.offset + self.l1*cos(q1) + self.l2*cos(q1+q2) + self.l3*cos(q1+q2+q3)
        x = self.l1*sin(q1) + self.l2*sin(q1+q2) + self.l3*sin(q1+q2+q3) 

        p = np.array([[z], [x]])

        # Task Jacobian: p_dot = Jp(q)q_dot
        Jp = np.array([[- self.l2*sin(q1 + q2) - self.l1*sin(q1) - self.l3*sin(q1 + q2 + q3), - self.l2*sin(q1 + q2) - self.l3*sin(q1 + q2 + q3), -self.l3*sin(q1 + q2 + q3)], \
                    [self.l2*cos(q1 + q2) + self.l1*cos(q1) + self.l3*cos(q1 + q2 + q3),   self.l2*cos(q1 + q2) + self.l3*cos(q1 + q2 + q3),  self.l3*cos(q1 + q2 + q3)]])
        
        # Distance from origin d
        d = sqrt(z**2 + x**2)

        # Extended jacobian for distance: d_dot = Jd(q)q_dot
        Jd = 1/d*p.transpose().dot(Jp)

        # Extended jacobian for distance: d_dot = Jd(q)q_dot
        Jinv = np.linalg.pinv(Jd)

        # Proportional Gain
        K = 1.4

        # Distance measured from WeArt 
        dr = distance
        dr_dot = 0        
        
        """
        l = len(d_pre)
        if(l == 6):
            dr_dot = 147/60*(distance - 360/147*d_pre[l-1] + 450/147*d_pre[l-2] - 400/147*d_pre[l-3] + 225/147*d_pre[l-4] - 72/147*d_pre[l-5] + 10/147*d_pre[l-6])/timeStep
        """
        pos = self.sim.getObjectPosition(self.Tip)
        x_sens = pos[0]
        y_sens = pos[2]
        
        self.filt.predict()
        self.filt.update(np.array([[x_sens],[y_sens]]))
      
        dr_dot = sqrt(self.filt.x[1,0]**2 + self.filt.x[3,0]**2)
        
        print(dr_dot)
        
       

        w = np.array([(pi/4-q1)/(pi/2), (pi/3-q2)/(2/3*pi), (pi/4-q3)/(pi/2)])
        u_vinc = (np.eye(3) - Jinv.dot(Jd))*w.transpose()
        qdot = Jinv.dot(dr_dot + K*(dr-d)) + u_vinc

        v1 = qdot[0][0]
        v2 = qdot[1][0]
        v3 = qdot[2][0]
    
        self.sim.setJointTargetVelocity(self.J1, v1)
        self.sim.setJointTargetVelocity(self.J2, v2)
        self.sim.setJointTargetVelocity(self.J3, v3)

        self.sim.setGraphStreamValue(self.graphHandle, self.graphD,d)
        self.sim.setGraphStreamValue(self.graphHandle, self.graph_ref,dr)

    def get_contact_force(self):
        return self.force_calculator.compute()  
        
    def start_simulation(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
    
    def step_simulation(self):
        self.sim.step()
    
    def stop_simulation(self):
        self.sim.removeDrawingObject(self.end_trace)
        self.sim.destroyGraphCurve(self.graphHandle,-1)
        self.sim.stopSimulation()


class SensorForceCalculator:
    MAX_FORCE = 1

    def __init__(self, copp: CoppeliaConnector):
        self.copp = copp
    
    def compute(self):        
        for i in range(0,1):
            _, _, rForce, n = self.copp.sim.getContactInfo(self.copp.sim.handle_all, self.copp.L3, i)
            if rForce == []:
                break # no more contacts
        if len(rForce) != 0:
            Fx = rForce[0]
            Fz = rForce[2]
            total_force = sqrt(Fx**2 + Fz**2)
        else:
            total_force = 0
        return total_force