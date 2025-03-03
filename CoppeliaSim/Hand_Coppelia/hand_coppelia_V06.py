from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from math import sqrt 
import numpy as np
import sympy as sym
from thumb_sim_V06 import *
from index_sim_V06 import *
from middle_sim_V06 import *
from weart import *

class SceneException(Exception):
    pass

class CoppeliaConnector:

    LIM_INDEX = np.radians([75, 60, 60])
    LIM_MIDDLE = np.radians([75, 60, 60])
    LIM_THUMB = np.radians([75, 80])
    INTEGRAL = 0

    def __init__(self):
        self._client = RemoteAPIClient()
        self.sim = self._client.require("sim")

        self._check_correct_engine()
        
        self._fetch()
        
        self._initialize_symbolic_task()

        self._initialize_maxmin_dist()

        self.force_calculator = SensorForceCalculator(self)

    def _check_correct_engine(self):
        engine = self.sim.getInt32Param(self.sim.intparam_dynamic_engine)
        if engine != self.sim.physics_mujoco:
            raise SceneException("The scene must run using the MuJoCo physics engine.")

    def _fetch(self):
        # Handles for hand joints and force "sensors"
        self.thumb_1 = self.sim.getObject("/Hand/Thumb_1")
        self.thumb_2 = self.sim.getObject("/Hand/Thumb_2")
        self.thumb_tip = self.sim.getObject("/Hand/Thumb_Tip")
        self.index_1 = self.sim.getObject("/Hand/Index_1")
        self.index_2 = self.sim.getObject("/Hand/Index_2")
        self.index_3 = self.sim.getObject("/Hand/Index_3")
        self.index_tip = self.sim.getObject("/Hand/Index_Tip")
        self.middle_1 = self.sim.getObject("/Hand/Middle_1")
        self.middle_2 = self.sim.getObject("/Hand/Middle_2")
        self.middle_3 = self.sim.getObject("/Hand/Middle_3")
        self.middle_tip = self.sim.getObject("/Hand/Middle_Tip")
        self.ring_1 = self.sim.getObject("/Hand/Ring_1")
        self.ring_2 = self.sim.getObject("/Hand/Ring_2")
        self.ring_3 = self.sim.getObject("/Hand/Ring_3")
        self.ring_tip = self.sim.getObject("/Hand/Ring_Tip")
        self.pinky_1 = self.sim.getObject("/Hand/Pinky_1")
        self.pinky_2 = self.sim.getObject("/Hand/Pinky_2")
        self.pinky_3 = self.sim.getObject("/Hand/Pinky_3")
        self.pinky_tip = self.sim.getObject("/Hand/Pinky_Tip")

        self.force_index = self.sim.getObject("/Hand/Force_Index")
        self.force_middle = self.sim.getObject("/Hand/Force_Middle")
        self.force_thumb = self.sim.getObject("/Hand/Force_Thumb")

        # Graphs used to compare reference distance and tracked distance
        self.graphHandleIndex = self.sim.getObject("/Graph_Index")
        self.graphDIndex = self.sim.addGraphStream(self.graphHandleIndex, "Distance", "", 0)
        self.graph_refIndex = self.sim.addGraphStream(self.graphHandleIndex, "Reference", "", 0, [0.5, 1, 0])

        self.graphHandleMiddle = self.sim.getObject("/Graph_Middle")
        self.graphDMiddle = self.sim.addGraphStream(self.graphHandleMiddle, "Distance", "", 0)
        self.graph_refMiddle = self.sim.addGraphStream(self.graphHandleMiddle, "Reference", "", 0, [0.5, 1, 0]) 

        self.graphHandleThumb = self.sim.getObject("/Graph_Thumb")
        self.graphDThumb = self.sim.addGraphStream(self.graphHandleThumb, "Distance", "", 0)
        self.graph_refThumb = self.sim.addGraphStream(self.graphHandleThumb, "Reference", "", 0, [0.5,1,0]) 

    # Calculate symbolic task
    def _initialize_symbolic_task(self):
        # Definition of symbols
        self.j1, self.j2, self.j3 = sym.symbols("j1 j2 j3", real = True)

        # Positions of thumb's joints with respect to the previous one
        p0_1 = np.array(self.sim.getObjectPosition(self.thumb_1))
        p1_2 = self.sim.getObjectPosition(self.thumb_2, self.thumb_1)
        p2_t = self.sim.getObjectPosition(self.thumb_tip, self.thumb_2)

        # Euler angles of thumb's joints with respect to the previous one
        o0_1 = np.array(self.sim.getObjectOrientation(self.thumb_1))
        o1_2 = np.array(self.sim.getObjectOrientation(self.thumb_2, self.thumb_1))

        self.pos_thumb, _ = symbolic_task_thumb(np.array([p0_1, p1_2, p2_t]), [o0_1, o1_2], [self.j1, self.j2])

        # Positions of index's joints with respect to the previous one
        p0_1 = self.sim.getObjectPosition(self.index_1)
        p1_2 = self.sim.getObjectPosition(self.index_2, self.index_1)
        p2_3 = self.sim.getObjectPosition(self.index_3, self.index_2)
        p3_t = self.sim.getObjectPosition(self.index_tip, self.index_3)
        # Euler angles of index's joints with respect to the previous one
        o0_1 = np.array(self.sim.getObjectOrientation(self.index_1))
        o1_2 = np.array(self.sim.getObjectOrientation(self.index_2, self.index_1))
        o2_3 = np.array(self.sim.getObjectOrientation(self.index_3, self.index_2))
                
        self.pos_index, _ = symbolic_task_index(np.array([p0_1, p1_2, p2_3, p3_t]), [o0_1, o1_2, o2_3], [self.j1, self.j2, self.j3])

        # Positions of middle's joints with respect to the previous one
        p0_1 = np.array(self.sim.getObjectPosition(self.middle_1))
        p1_2 = self.sim.getObjectPosition(self.middle_2, self.middle_1)
        p2_3 = self.sim.getObjectPosition(self.middle_3, self.middle_2)
        p3_t = self.sim.getObjectPosition(self.middle_tip, self.middle_3)
        # Euler angles of middle's joints with respect to the previous one
        o0_1 = np.array(self.sim.getObjectOrientation(self.middle_1))
        o1_2 = np.array(self.sim.getObjectOrientation(self.middle_2, self.middle_1))
        o2_3 = np.array(self.sim.getObjectOrientation(self.middle_3, self.middle_2))

        self.pos_middle, _ = symbolic_task_middle(np.array([p0_1, p1_2, p2_3, p3_t]), [o0_1, o1_2, o2_3], [self.j1, self.j2, self.j3])

    # Calculate maximum and minimum distance for mapping
    def _initialize_maxmin_dist(self):
        # Maximum and minimum distance for the thumb
        p = sym.matrices.dense.matrix2numpy(self.pos_thumb.subs([(self.j1, 0), (self.j2, 0)]))[0]
        p = np.reshape(p, (3, 1)).astype(float)
        self.max_dist_thumb = np.linalg.norm(p)
        p = sym.matrices.dense.matrix2numpy(self.pos_thumb.subs([(self.j1, self.LIM_THUMB[0]), (self.j2, self.LIM_THUMB[1])]))[0]
        p = np.reshape(p, (3, 1)).astype(float)
        self.min_dist_thumb = np.linalg.norm(p)

        # Maximum and minimum distance for the index
        p = sym.matrices.dense.matrix2numpy(self.pos_index.subs([(self.j1, 0), (self.j2, 0), (self.j3, 0)]))[0]
        p = np.reshape(p, (3, 1)).astype(float)
        self.max_dist_index = np.linalg.norm(p)
        p = sym.matrices.dense.matrix2numpy(self.pos_index.subs([(self.j1, self.LIM_INDEX[0]), (self.j2, self.LIM_INDEX[1]), (self.j3, self.LIM_INDEX[2])]))[0]
        p = np.reshape(p, (3, 1)).astype(float)
        self.min_dist_index = np.linalg.norm(p)

        # Maximum and minimum distance for the middle
        p = sym.matrices.dense.matrix2numpy(self.pos_middle.subs([(self.j1, 0), (self.j2, 0), (self.j3, 0)]))[0]
        p = np.reshape(p, (3, 1)).astype(float)
        self.max_dist_middle = np.linalg.norm(p)
        p = sym.matrices.dense.matrix2numpy(self.pos_middle.subs([(self.j1, self.LIM_MIDDLE[0]), (self.j2, self.LIM_MIDDLE[1]), (self.j3, self.LIM_MIDDLE[2])]))[0]
        p = np.reshape(p, (3, 1)).astype(float)
        self.min_dist_middle = np.linalg.norm(p)
                
    # Maps closure value, that is between 0 and 1, in the range [min_dist, max_dist]
    def mapping(self, info):
        max_dist = 0
        min_dist = 0
        closure = info[0]
        finger = info[1]
        match finger:
            case "thumb":
                max_dist = self.max_dist_thumb
                min_dist = self.min_dist_thumb
            case "index":
                max_dist = self.max_dist_index
                min_dist = self.min_dist_index
            case "middle":
                max_dist = self.max_dist_middle
                min_dist = self.min_dist_middle
            case _:
                max_dist = 0
        distance = max_dist - closure*(max_dist - min_dist)
        return distance
    

    def actuation(self, ref_distance, finger, weart: WeartConnector):
        match finger:
            case "thumb":
                # Get current joint positions
                q1 = self.sim.getJointPosition(self.thumb_1)
                q2 = self.sim.getJointPosition(self.thumb_2)

                # Inverse kinematics
                qdot, tip_distance = actuation_thumb(ref_distance, [q1, q2], self.LIM_THUMB)

                v1 = qdot[0][0]
                v2 = qdot[1][0]

                # Set the target velocities
                self.sim.setJointTargetVelocity(self.thumb_1, v1)
                self.sim.setJointTargetVelocity(self.thumb_2, v2)

                # For Graph
                self.sim.setGraphStreamValue(self.graphHandleThumb, self.graphDThumb, tip_distance)
                self.sim.setGraphStreamValue(self.graphHandleThumb, self.graph_refThumb, ref_distance)

            case "index": 
                # Get current joint positions
                q1 = self.sim.getJointPosition(self.index_1)
                q2 = self.sim.getJointPosition(self.index_2)
                q3 = self.sim.getJointPosition(self.index_3)

                # Inverse kinematics
                qdot, tip_distance, integral_updated = actuation_index(ref_distance, [q1, q2, q3], self.LIM_INDEX, self.INTEGRAL)

                v1 = qdot[0][0]
                v2 = qdot[1][0]
                v3 = qdot[2][0]
                
                self.INTEGRAL = integral_updated

                # Set the target velocities
                self.sim.setJointTargetVelocity(self.index_1, float(v1))
                self.sim.setJointTargetVelocity(self.index_2, float(v2))
                self.sim.setJointTargetVelocity(self.index_3, float(v3))

                # For Graph
                self.sim.setGraphStreamValue(self.graphHandleIndex, self.graphDIndex, tip_distance)
                self.sim.setGraphStreamValue(self.graphHandleIndex, self.graph_refIndex, ref_distance)
            
            case "middle": 
                # Get current joint positions
                q1 = self.sim.getJointPosition(self.middle_1)
                q2 = self.sim.getJointPosition(self.middle_2)
                q3 = self.sim.getJointPosition(self.middle_3)

                # Inverse kinematics
                qdot, tip_distance = actuation_middle(ref_distance, [q1, q2, q3], self.LIM_MIDDLE)

                v1 = qdot[0][0]
                v2 = qdot[1][0]
                v3 = qdot[2][0]

                # Set the target velocities
                self.sim.setJointTargetVelocity(self.middle_1, v1)
                self.sim.setJointTargetVelocity(self.middle_2, v2)
                self.sim.setJointTargetVelocity(self.middle_3, v3)

                # Set same target velocities as middle ones for ring and pinky
                self.sim.setJointTargetVelocity(self.ring_1, v1)
                self.sim.setJointTargetVelocity(self.ring_2, v2)
                self.sim.setJointTargetVelocity(self.ring_3, v3)

                self.sim.setJointTargetVelocity(self.pinky_1, v1)
                self.sim.setJointTargetVelocity(self.pinky_2, v2)
                self.sim.setJointTargetVelocity(self.pinky_3, v3)
                
                # For Graph
                self.sim.setGraphStreamValue(self.graphHandleMiddle, self.graphDMiddle, tip_distance)
                self.sim.setGraphStreamValue(self.graphHandleMiddle, self.graph_refMiddle, ref_distance)
            

        # Application of contact force
        force = self.force_calculator.compute(finger)
        weart.apply_force(force, finger)
    
    def start_simulation(self):
        self.sim.setStepping(True)
        self.sim.startSimulation()
    
    def step_simulation(self):
        self.sim.step()
    
    def stop_simulation(self):
        self.sim.destroyGraphCurve(self.graphHandleIndex, -1)
        self.sim.destroyGraphCurve(self.graphHandleMiddle, -1)
        self.sim.destroyGraphCurve(self.graphHandleThumb, -1)

        self.sim.stopSimulation()

class SensorForceCalculator:
    MAX_FORCE_IND = 7
    MAX_FORCE_MID = 4  
    MAX_FORCE_THU = 4

    def __init__(self, copp: CoppeliaConnector):
        self.copp = copp
    
    def compute(self, finger):
        MAX_FORCE = 0
        match finger:
            case "thumb":
                MAX_FORCE = self.MAX_FORCE_THU
                for i in range(0,1):
                    _, _, rForce, _ = self.copp.sim.getContactInfo(self.copp.sim.handle_all, self.copp.force_thumb, i)
                    if rForce == []:
                        break # no more contacts
            case "index":
                MAX_FORCE = self.MAX_FORCE_IND
                for i in range(0,1):
                    _, _, rForce, _ = self.copp.sim.getContactInfo(self.copp.sim.handle_all, self.copp.force_index, i)
                    if rForce == []:
                        break # no more contacts
            case "middle":
                MAX_FORCE = self.MAX_FORCE_MID
                for i in range(0,1):
                    _, _, rForce, _ = self.copp.sim.getContactInfo(self.copp.sim.handle_all, self.copp.force_middle, i)
                    if rForce == []:
                        break # no more contacts
        # Calculate norm of the force vector
        if len(rForce) != 0:
            Fx = rForce[0]
            Fy = rForce[1]
            Fz = rForce[2]
            total_force = sqrt(Fx**2 + Fy**2 + Fz**2)
        else:
            total_force = 0
        return total_force/MAX_FORCE



    

