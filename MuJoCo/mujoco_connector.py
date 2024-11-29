from interfaces import Engine, Visualizer
from hand import Hand
from math import radians

from math import pi
import mujoco as mj
import mujoco.viewer as mj_viewer

import numpy as np
from thumb_sim import *
from index_sim import *
from middle_sim import *
from annular_sim import *
from pinky_sim import *

class MujocoConnector(Engine):
    def __init__(self, xml_path: str, hands: tuple[Hand, Hand]):
        """Creates the MuJoCo Connector with the MJCF at the passed path.

        Args:
            xml_path (str): path to the XML file containing the MJCF
            hands (tuple[Hand, Hand]): hands configuration
        """
        spec = mj.MjSpec()
        spec.from_file(xml_path)

        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        opt = mj.MjvOption()
        self._fetch_hands(hands)
        self._fetch_finger_joints()
        self._fetch_collidable_object()

        self._edit_hands(spec, hands)

        self._should_reset = False
        self.model.opt.timestep = 0.001

        mj.mj_forward(self.model, self.data)

    def _edit_hands(self, spec: mj.MjSpec, hands: tuple[Hand, Hand]):
        for hand in hands:
            hand_body = spec.find_body(f"{hand.side.capitalize()}_Free")

            if hand.tracking or hand.haptics:
                controller_rotation = hand.controller_rotation if spec.degree else radians(hand.controller_rotation)
                hand_body_rotation = hand_body.alt
                
                hand_body_rotation.euler[1] += controller_rotation
                hand_body.alt = hand_body_rotation
                
            else:
                # if this hand is used for neither tracking nor haptics, we can delete it from the scene.
                spec.detach_body(hand_body)
                # detach takes care of removing weld constraints, sensors and everything.
    
    def _get_flex_id(self, flex_name: str):
        for id, name_adr in enumerate(self.model.name_flexadr):
            name_binary: bytes = self.model.names[name_adr:]
            name_decoded = name_binary.decode()
            name_decoded = name_decoded[:name_decoded.index("\0")]
            if name_decoded == flex_name:
                return id
        return None

    # Define all the collidable object for which you want a texture and temperature haptic feedback
    def _fetch_collidable_object(self):
        self.coll_object = [self.data.geom("cube_silver_coll").id, self.data.geom("cylinder_plastic_coll").id, self.data.geom("cube_rock_coll").id, self.data.geom("sphere_plastic_coll").id, self._get_flex_id("left kidney") - 1]
        self.TEMPERATURE_TEXTURE = {self.coll_object[0] : [0.3, 3], self.coll_object[1] : [0.5, 5], self.coll_object[2] : [0.4, 10], self.coll_object[3] : [0.5, 5], self.coll_object[4] : [0.7, 14]}
        # Texture 
        # 20 - Aluminium, 5 - PlasticMeshSlow, 10 - CrushedRock, 14 - ProfiledRubberSlow

    def _fetch_hands(self, hands: tuple[Hand, Hand]):
        self._hand_mocaps = [self.model.body(f"{hand.side}_hand_mocap").mocapid[0] if hand.tracking else 0 for hand in hands]
    
    def _fetch_finger_joints(self):
        self.joint_ids ={}
        for hand in ["Right_", "Left_"]:
            for finger in ["Thumb_", "Index_", "Middle_", "Annular_", "Pinky_"]:
                for joint in ["J1", "J2", "J3"]:
                    name = hand + finger + joint
                    self.joint_ids[name] = self.model.actuator(name + ".stl").id

    # Track position and orientation of Oculus controller
    def move_hand(self, hand_id: int, position: list[float], rotation: list[float]):
        self.data.mocap_pos[self._hand_mocaps[hand_id]] = position
        self.data.mocap_quat[self._hand_mocaps[hand_id]] = rotation #quaternion_multiply(rotation, [ 0.5003982, 0.4996018, -0.4999998, -0.4999998 ])

    def init_task(self):
        # _______________ Thumb ___________________________________________
        L12 = self.data.xpos[self.model.body("Left_Thumb_J2.stl").id] - self.data.xpos[self.model.body("Left_Thumb_J1.stl").id]
        L23 = self.data.xpos[self.model.body("Left_Thumb_J3.stl").id] - self.data.xpos[self.model.body("Left_Thumb_J2.stl").id]
        L3T = self.data.site_xpos[self.model.site("Left_forSensorThumb_3.stl").id] - self.data.xpos[self.model.body("Left_Thumb_J3.stl").id]

        self.thumb_links = [np.linalg.norm(L12), np.linalg.norm(L23), np.linalg.norm(L3T)]
        self.thumb_offset = np.linalg.norm(self.data.xpos[self.model.body("Left_Thumb_1.stl").id] - self.data.xpos[self.model.body("Left_Middle_1.stl").id]) 

        # _______________ Index ___________________________________________
        L12 = self.data.xpos[self.model.body("Left_Index_J2.stl").id] - self.data.xpos[self.model.body("Left_Index_J1.stl").id]
        L23 = self.data.xpos[self.model.body("Left_Index_J3.stl").id] - self.data.xpos[self.model.body("Left_Index_J2.stl").id]
        L3T = self.data.site_xpos[self.model.site("Left_forSensor").id] - self.data.xpos[self.model.body("Left_Index_J3.stl").id]
        
        self.index_links = [np.linalg.norm(L12), np.linalg.norm(L23), np.linalg.norm(L3T)]
        self.index_offset = np.linalg.norm(self.data.xpos[self.model.body("Left_Index_1.stl").id] - self.data.xpos[self.model.body("Left_Index_J1.stl").id])

        # _______________ Middle ___________________________________________
        L12 = self.data.xpos[self.model.body("Left_Middle_J2.stl").id] - self.data.xpos[self.model.body("Left_Middle_J1.stl").id]
        L23 = self.data.xpos[self.model.body("Left_Middle_J3.stl").id] - self.data.xpos[self.model.body("Left_Middle_J2.stl").id]
        L3T = self.data.site_xpos[self.model.site("Left_forSensorMiddle_4.stl").id] - self.data.xpos[self.model.body("Left_Middle_J3.stl").id]

        self.middle_links = [np.linalg.norm(L12), np.linalg.norm(L23), np.linalg.norm(L3T)]
        self.middle_offset = np.linalg.norm(self.data.xpos[self.model.body("Left_Middle_1.stl").id] - self.data.xpos[self.model.body("Left_Middle_J1.stl").id]) 
        
        # _______________ Annular ___________________________________________
        L12 = self.data.xpos[self.model.body("Left_Annular_J2.stl").id] - self.data.xpos[self.model.body("Left_Annular_J1.stl").id]
        L23 = self.data.xpos[self.model.body("Left_Annular_J3.stl").id] - self.data.xpos[self.model.body("Left_Annular_J2.stl").id]
        L3T = self.data.site_xpos[self.model.site("Left_forSensorAnnular_4.stl").id] - self.data.xpos[self.model.body("Left_Annular_J3.stl").id]

        self.annular_links = [np.linalg.norm(L12), np.linalg.norm(L23), np.linalg.norm(L3T)]
        self.annular_offset = np.linalg.norm(self.data.xpos[self.model.body("Left_Annular_1.stl").id] - self.data.xpos[self.model.body("Left_Annular_J1.stl").id])
        # _______________ Pinky ___________________________________________
        L12 = self.data.xpos[self.model.body("Left_Pinky_J2.stl").id] - self.data.xpos[self.model.body("Left_Pinky_J1.stl").id]
        L23 = self.data.xpos[self.model.body("Left_Pinky_J3.stl").id] - self.data.xpos[self.model.body("Left_Pinky_J2.stl").id]
        L3T = self.data.site_xpos[self.model.site("Left_forSensorPinky_4.stl").id] - self.data.xpos[self.model.body("Left_Pinky_J3.stl").id]

        self.pinky_links = [np.linalg.norm(L12), np.linalg.norm(L23), np.linalg.norm(L3T)]
        self.pinky_offset = np.linalg.norm(self.data.xpos[self.model.body("Left_Pinky_1.stl").id] - self.data.xpos[self.model.body("Left_Pinky_J1.stl").id])

        # _______________ Joint Limits (for Null Space control) ___________
        self.index_qmin = self.middle_qmin = self.annular_qmin = self.pinky_qmin = self.thumb_qmin = 0
        self.index_qmax = self.middle_qmax = self.annular_qmax = self.pinky_qmax = pi/2
        self.thumb_qmax = [pi/6, pi/2, pi/2]

    # Maps WeArt closure parameter between maximum and minimum Task Distance
    def mapping(self, closure, finger):
        max_dist = 0
        min_dist = 0
        match finger:
            case "thumb":
                max_dist = 0.0756
                min_dist = 0.005
            case "thumb_abd":
                max_dist = 0.0
                min_dist = -0.07
            case "index": 
                max_dist = 0.07
                min_dist = 0.01
            case "middle": 
                max_dist = 0.083
                min_dist = 0.01
            case "annular": 
                max_dist = 0.076
                min_dist = 0.01
            case "pinky": 
                max_dist = 0.053
                min_dist = 0.01
        distance = max_dist - closure*(max_dist - min_dist)
        return distance

    def move_finger(self, hand_id: int, finger: str, closure: float, abduction: float):
        
        if hand_id == 1:
            hand = "Right_"
            code_off = 7 # offset for data struct to get access to right fingers
        else: 
            hand = "Left_"
            code_off = 14 # offset for data struct to get access to left fingers

        dr = self.mapping(closure, finger)

        # Get joint id
        J1 = self.joint_ids[hand + finger.capitalize() + "_J1"]
        J2 = self.joint_ids[hand + finger.capitalize() + "_J2"]
        J3 = self.joint_ids[hand + finger.capitalize() + "_J3"]

        # Get joint value
        q1 = self.data.qpos[J1 + code_off]
        q2 = self.data.qpos[J2 + code_off]
        q3 = self.data.qpos[J3 + code_off]

        q = [q1, q2, q3]
        
        # Inverse Kinematics for each finger
        match finger:
            case "thumb":
                v1,v2,v3 = move_thumb(dr,q,self.thumb_links,self.thumb_offset,self.thumb_qmin,self.thumb_qmax)
            case "index":
                v1,v2,v3 = move_index(dr,q,self.index_links,self.index_offset,self.index_qmin,self.index_qmax)
            case "middle":
                v1,v2,v3 = move_middle(dr,q,self.middle_links,self.middle_offset,self.middle_qmin,self.middle_qmax)
            case "annular":
                v1,v2,v3 = move_annular(dr,q,self.annular_links,self.annular_offset,self.annular_qmin, self.annular_qmax)
            case "pinky":
                v1,v2,v3 = move_pinky(dr,q,self.pinky_links,self.pinky_offset,self.pinky_qmin,self.pinky_qmax)

        # Apply target velocities to actuators
        self.data.ctrl[J1] = v1
        self.data.ctrl[J2] = v2
        self.data.ctrl[J3] = v3

    def get_contact_force(self, hand_id: int, finger: str) -> float:
        temperature = 0.5 # Room temperature
        texture = 0 # Default texture - ClickNormal

        hand_side = "Left" if hand_id == 0 else "Right"
        sensor_name = hand_side + "_fingertip_sensor_" + finger 
        data = self.data.sensor(sensor_name).data

        # Since collision box is bigger than sensor site, we filter case in which there is a "real contact"
        force = data[0] / 200

        if (force > 0.01):
            # fetch collision geom relative to last phalanx
            if finger == "thumb":
                name_geom1 = hand_side + "_" + finger.capitalize() + "_3.stl_collision"
            else: 
                name_geom1 = hand_side + "_" + finger.capitalize() + "_4.stl_collision"
            geom_sens = self.data.geom(name_geom1).id

            for i in range(self.data.ncon):
                # iterate over each contact looking for one that involves the last phalanx and a collidable objcet
                contact = self.data.contact[i]
                geom1_id = contact.geom1
                geom2_id = contact.geom2

                # Check if either geom1 or geom2 is the specified geom
                if (geom1_id == geom_sens and geom2_id in self.coll_object):
                    temperature, texture = self.TEMPERATURE_TEXTURE[geom2_id]
        else:
            force = 0 
        return force, temperature, texture

    def step_simulation(self, duration: float | None):
        if self._should_reset:
            mj.mj_resetData(self.model, self.data)
            self._should_reset = False

        if duration is None:
            mj.mj_step(self.model, self.data)
        else:
            step_count = int(duration // self.model.opt.timestep)
            for _ in range(step_count):
                mj.mj_step(self.model, self.data)
    
    def reset_simulation(self):
        self._should_reset = True

class MujocoSimpleVisualizer(Visualizer):
    def __init__(self, mujoco: MujocoConnector, framerate: int | None = None):
        self._mujoco = mujoco
        self._scene = mj.MjvScene(mujoco.model, 1000)
        self._framerate = framerate
    
    def start_visualization(self):
        self._viewer = mj_viewer.launch_passive(self._mujoco.model, self._mujoco.data,
                                                show_left_ui=False, show_right_ui=False)
        self._viewer.cam.azimuth = 138
        self._viewer.cam.distance = 3
        self._viewer.cam.elevation = -16

    def render_frame(self):
        self._viewer.sync()

    def should_exit(self):
        return not self._viewer.is_running()
    
    def stop_visualization(self):
        self._viewer.close()