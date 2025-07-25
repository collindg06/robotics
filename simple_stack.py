# tbd: current mode: i, o, direct joint offset, i for 6 finger, o for first three joint
#   intend to use hand_off for end effort, which would need rpmflow (ik solver)
#   see target_follow example.

from omni.isaac.core.tasks import BaseTask

from .ur10ext import BaseSample
from omni.isaac.ur10hand.controllers.stacking_controller import StackingController
import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.objects import DynamicCapsule
import omni.usd
from pxr import UsdPhysics, UsdGeom
from pxr import UsdPhysics
from omni.isaac.ur10hand import UR10 
from omni.isaac.franka import Franka 
from omni.isaac.core.utils.types import ArticulationAction
import carb
import omni
import weakref
import os, sys
import torch
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.ur10hand.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
import asyncio
import json
import websockets
import time
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_id = ext_manager.get_extension_id_by_module(__name__)
ext_path = ext_manager.get_extension_path(ext_id)
print('ext path', ext_path)
print('xxxxx torch ', torch.__version__)

#sys.path.append("/home/student/Documents/IsaacLab/source/extensions/omni.isaac.lab")
#from omni.isaac.lab.devices import Se3Keyboard
#from omni.isaac.franka.tasks import Stacking

class UR10Playing(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return
    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_extension_id_by_module(__name__)
        ext_path = ext_manager.get_extension_path(ext_id)
        print('ext path', ext_path)        
        scene.add_default_ground_plane()
        '''
        self._cylinder = scene.add(DynamicCylinder(prim_path="/World/red_cylinder",
                                            name="red_cylinder",
                                            position=np.array([0.55, 0.5, 0.45]),
                                            scale=np.array([0.02, 0.02, 0.1]),
                                            color=np.array([1.0, 0.0, 0.0])))
        
        stage = omni.usd.get_context().get_stage()
        prim_path = "/World/red_cylinder"
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise RuntimeError(f"Prim at {prim_path} not found!")

        # Apply Collision API to the main cylinder prim if not already present
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        #(Advanced) Create a child prim under it to act as the physics shape
        collision_prim_path = f"{prim_path}/PhysicsShape_Collision"

        # Define a Cylinder as the explicit physics shape
        collision_prim = stage.DefinePrim(collision_prim_path, "Cylinder")

        #  Set radius and height (match your visual shape)
        cylinder_geom = UsdGeom.Cylinder(collision_prim)

        # Optionally customize the collider size (match your visual model scale)
        cylinder_geom.CreateRadiusAttr(0.02)  # Match your DynamicCylinder radius
        cylinder_geom.CreateHeightAttr(0.1)   # Match your DynamicCylinder height

        
        self._capsule = scene.add(DynamicCapsule(prim_path="/World/my_dynamic_capsule",
                                            name="my_dynamic_capsule",
                                            position=np.array([0.5, 0.5, 0.45]),  # XYZ position (meters)
                                            scale=np.array([0.02, 0.02, 0.1]),   # X/Y = radius, Z = half-height
                                            color=np.array([0.8, 0.2, 0.2])))
        '''
        # Add 2 cubes to scene
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.5, 0.5, 0.4]),
                                            orientation=np.array([0.96592584,0., 0., 0.25881901]),
                                            scale=np.array([0.03, 0.03, 0.0415]),
                                            color=np.array([0, 0, 1.0])))
        
        self._cube2 = scene.add(DynamicCuboid(prim_path="/World/ref_cube",
                                            name="ref_cube",
                                            position=np.array([0.1, 0.1, 0.1]),
                                            orientation=np.array([0.96592584,0., 0., 0.25881901]),#(wxyz) 30 deg z-axis, don't use x-axis here since it will fall back to straight up-right after gravity 
                                            scale=np.array([0.0415, 0.0415, 0.0415]),
                                            color=np.array([0, 1.0, .0])))
        #exts/omni.isaac.universal_robots/omni/isaac/universal_robots/tasks/bin_filling.py
        self._ur10_asset_path = "C:\\isaac-sim\\exts\\ur10hand_demo.ur10ext\\config\\ur10_bin_filling2.usd"
        add_reference_to_stage(usd_path=self._ur10_asset_path, prim_path="/World/Scene")
        self._franka = scene.add(UR10(prim_path="/World/Scene/ur10",
                                        name="fancy_franka",
                                        position=np.array([0., 0., 0.5])))
        # scene.add add a robot obj to whatever is @prim_path, it must match the sub prim if the prim is not empty

        return

    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations



class SimpleStack(BaseSample):
    def __init__(self, exthandle=None) -> None:
        super().__init__()
        self.exthandle=exthandle # exthandle passed from simple_stack_extension
        self.dbg=False
        self.ik_conti=False
        self._controller = None
        self._articulation_controller = None
        self.finger_off=[0.,0.,0.,0.,0., 0., 0.,0.,0.,0.]
        self.arm_off=[0.,0.,0.,0.,0., 0.]
        self.hand_off=[0.,0.,0.]
        self.root_pose_0=torch.tensor([[0., 0., 0.],[0.,0.,0.]], device='cuda')
        self.pose_flag=False
        self.f_indx=1
        self.handmode='i'
        self.multiple = False
        self.multiplem = False
        self.z = False
        self.x = [0,0,0,0,0,0]
        self.limit_thumb = 0
        self.limiter = 0
        self.action_enable=True
        self.websocket=False
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input= carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args: self._on_keyboard_event(event, *args),
            )

    def setup_scene(self):
        self._world = self.get_world()
        self._world.add_task(UR10Playing(name="stacking_task"))
        print('robots ', self._world.scene._scene_registry.robots)

        return

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            print('ur10 keyboard event', event.input.name)
            if event.input.name == "KEY_0":
                self.f_indx=0
            if event.input.name == "KEY_1":
                self.f_indx=1
            if event.input.name == "KEY_2":
                self.f_indx=2
            if event.input.name == "KEY_3":
                self.f_indx=3
            if event.input.name == "KEY_4":
                self.f_indx=4
            if event.input.name == "KEY_5":
                self.f_indx=5
            if event.input.name == "KEY_7": #hand orientation rot x
                self.f_indx=7
            if event.input.name == "KEY_8": #hand orientation rot y 
                self.f_indx=9
            if event.input.name == "KEY_9":
                #reset all to 0
                for i in range(6):
                    self.finger_off[i]=0.
            if event.input.name == "I":
                self.handmode='i'
            if event.input.name == "O":
                self.handmode='o'
            if event.input.name == "A":
                self.handmode='a'
            if event.input.name == "D":
                self.dbg=True
            if event.input.name == "C": #toggle apply action or not
                self.action_enable= not self.action_enable
                print(f"zzz action_enable = {self.action_enable}")
            if event.input.name == "P": #send websocket
                self.websocket= not self.websocket
                print(f"zzz websocket = {self.websocket}")
            # Stop Z
            if event.input.name == "X":
                self.z = False 
            # Pick block up
            if event.input.name == "Z":
                self.starttime = time.perf_counter()
                self.z = True
                self.pose_flag = True
            # Small addition to finger index of choice
            if self.handmode == 'i' and event.input.name == "U":
                self.finger_off[self.f_indx]= 0.05
                self.pose_flag=True
            # Small subtraction to finger index of choice
            if self.handmode == 'i' and event.input.name == "J":
                self.finger_off[self.f_indx]= -0.05
                self.pose_flag=True
            if self.handmode == 'i' and event.input.name == "B":
                self.multiple = True
                self.arm_off= [-0.05, -0.9, 2, -.9, 0.05, -1.95, 0,0,0,0,0,0]
                self.pose_flag=True
            # Hover in front of block    
            if self.handmode == 'i' and event.input.name == "M":
                self.multiple = True
                self.arm_off=[-.05,-1,2,-.6,-.1,-3.3,   0,0,0,0,0,0]
                self.pose_flag=True
            if self.handmode == 'i' and event.input.name == "N":
                self.multiplem = True
                self.arm_off=[0, -0.35, 0.4, -0.8, -0.05, -1.45, -1,0,0,0,1,0]
                self.pose_flag=True
            if self.handmode == 'i' and event.input.name == "L":
                self.multiple = True
                self.arm_off=[.25, -.95, 2.1, -1.05,.2, -3.5,   0,0,0.15,0,0,0 ]
                self.pose_flag=True
            # Close fingers
            if self.handmode == 'i' and event.input.name == "Y":
                if self.limit_thumb <= .30 :
                    self.multiplem = True
                    self.arm_off= [ 0, 0, 0, 0, 0, 0 ,   0,0,.05,.05,0,0]
                    self.pose_flag=True
                else:
                    print("past limit")
            # Open fingers
            if self.handmode == 'i' and event.input.name == "K":
                if self.limit_thumb >= .05:
                    self.multiplem = True
                    self.arm_off= [ 0, 0, 0, 0, 0, 0 ,  0,0,-.05,-.05,0,0]
                    self.pose_flag=True
                else:
                    print("past limit")
            # Hover over block
            if self.handmode == 'i' and event.input.name == "Q":
                self.multiple = True
                self.arm_off=[.25, -.95, 2.1, -.8,.15, -3.25,   0,0,0.15,0,0,0 ]
                self.pose_flag=True
            # Grab block
            if self.handmode == 'i' and event.input.name == "S":
                self.multiple = True
                self.arm_off=[.25, -.95, 2.1, -.8,.15, -3.25,   0,0,0.38,0.15,0,0 ]
                self.pose_flag=True
            # Lift arm up
            if self.handmode == 'i' and event.input.name == "T":
                self.multiplem = True
                self.arm_off= [ 0, -.1, 0, 0, 0, 0 ,0,0,0,0,0,0]
                self.pose_flag=True
            # Small addition to arm index of choice
            if self.handmode == 'a' and event.input.name == "U":
                self.arm_off[self.f_indx]= 0.05
                self.pose_flag=True
            # Small subtraction to arm index of choice
            if self.handmode == 'a' and event.input.name == "J":
                self.arm_off[self.f_indx]= -0.05
                self.pose_flag=True
            if self.handmode == 'a' and event.input.name == "M":
                self.multiple = True
                self.arm_off=[1.4,1.25,-1.65,0.,0., 0.]
                self.pose_flag=True
            if self.handmode == 'o' and event.input.name == "U":
                self.hand_off [0] = 0.0
                self.hand_off [1] = 0.0
                self.hand_off [2] = 0.0
                self.hand_off[self.f_indx]= 0.05
                self.pose_flag=True
            if self.handmode == 'o' and event.input.name == "MINUS":
                self.hand_off[self.f_indx]= 0.0
                self.pose_flag=True
            if self.handmode == 'o' and event.input.name == "J":
                self.hand_off = [0.0, 0.0, 0.0]
                self.hand_off[self.f_indx]= -0.05
                self.pose_flag=True
        return True

    async def send_nats_message(self, position, quat_target=None):
        uri = "wss://service.zenimotion.com/nats"
        # uri = "ws://134.209.218.187:8081/nats"
        x=float(position[0])
        y=float(position[1])
        z=float(position[2])
        if not quat_target is None:
            quat0= quat_target
        else:
            quat0=q0
        print("xxx ", x,y,z, quat0[0], quat0)
        msg_json = json.dumps({"position":{"z": z, "y": y, "x": x}, "orientation":{"x": float(quat0[0]), "w": float(quat0[3]), "y": float(quat0[1]), "z": float(quat0[2])}, "channel":"B11772DB-2A8B-4647-A9D3-3B6CD439350C"})
        msglen=len(msg_json)
        print(msg_json)
        async with websockets.connect(uri) as websocket:
            message = f"PUB subject.pose {msglen}\r\n{msg_json}\r\n"
            await websocket.send(message)
            print("Sent message to NATS WebSocket")
    async def setup_post_load(self):
        print('robots ', self._world.scene._scene_registry.robots)
        self._franka_task = self._world.get_task(name="stacking_task")
        #self._task_params = self._franka_task.get_params()
        print("xxx self.world ", self._world)
        self.my_franka = self._world.scene.get_object("fancy_franka") #object name is not the prim path
        self._cube = self._franka_task._cube
        print("xxx self.world ", self.my_franka)
        self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=self.my_franka)

        self._articulation_controller = self.my_franka.get_articulation_controller()
        #self.teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
        #self.teleop_interface.add_callback("L", reset_sim)

        #callback to start tang stuff after load the scene
        print('vvv exthandle ', self.exthandle)
        #self.exthandle.startup_tang()
        return

    def _on_stacking_physics_step(self, step_size):
        observations = self._world.get_observations()
        #bservation  {'fancy_franka': {'joint_positions': array([ 18 joints...], dtype=float32)}, 'fancy_cube': {'position': array([x,y,z], dtype=float32), 'goal_position': array([-0.3    , -0.3    ,  0.02575])}}
        #print('observation ', observations)
        ee_pose_w = self.my_franka._end_effector.get_world_pose() #tuple (pose, quat)
        cube_pose = self._cube.get_world_pose()
        if self.dbg:
            self.dbg=False
            print('xxx eepose ', ee_pose_w[0])
            print('xxx cube pose ', cube_pose)

        actions = ArticulationAction(joint_positions=np.array([0.0, -0.5, 0.54, 0. , 0.0, 0 , 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.]))
        actions.joint_positions=observations['fancy_franka']['joint_positions']
        if self.pose_flag and not self.handmode=='o' and not self.z == True:
            self.pose_flag=False
            # Check if moving multiple joints and want to add the changes on top of current position
            if self.handmode=='i' and self.multiplem == True :
                actions.joint_positions[0:12] =  actions.joint_positions[0:12] + self.arm_off
                self.multiplem = False
                self.limiter = 2
            # Check if moving multiple joints 
            elif self.handmode=='i' and self.multiple == True :
                self.limiter = 1
                actions.joint_positions[0:12] =  (actions.joint_positions[0:12] * 0) + self.arm_off
                self.multiple = False
            # If you just want to move 1 finger
            elif self.handmode=='i':#finger
                self.limit_thumb = self.limit_thumb + self.finger_off[self.f_indx]
                if self.f_indx != 2 :
                    actions.joint_positions[self.f_indx+6] +=self.finger_off[self.f_indx]
                elif (self.limit_thumb >= 0 and self.limit_thumb <= .35):
                    actions.joint_positions[self.f_indx+6] +=self.finger_off[self.f_indx]
                elif (self.limit_thumb < 0):
                    print("past limit.")
                    self.limit_thumb = 0
                elif (self.limit_thumb > .35):
                    print("past limit")
                    self.limit_thumb = .35
            # Check if moving multiple joints 
            if self.handmode=='a' and self.multiple == True :
                actions.joint_positions[0:6] =  (actions.joint_positions[0:6] * 0) + self.arm_off
                self.multiple = False
            elif self.handmode=='a':#arm joints
                actions.joint_positions[self.f_indx] +=self.arm_off[self.f_indx]
                self.x[self.f_indx] += self.arm_off[self.f_indx]

            # Apply action to robot
            self._articulation_controller.apply_action(actions)
            print("Mine: ", self.x)
            print("yyyxxx actions ", actions)

            # Updating limiter
            if self.limiter == 1:
                self.limit_thumb = self.arm_off[8]
                self.limiter = 0
            elif self.limiter == 2:
                self.limit_thumb = self.limit_thumb + self.arm_off[8]
                self.limiter = 0

        # Get to different positions every few seconds to pick up block
        elif self.pose_flag and self.z == True:
                elapsed = time.perf_counter() - self.starttime
                if (elapsed > 0 and elapsed < 3):
                    self.arm_off=[-.05,-1,2,-.6,-.1,-3.3,   0,0,0,0,0,0]
                    actions.joint_positions[0:12] =  (actions.joint_positions[0:12] * 0) + self.arm_off
                    self._articulation_controller.apply_action(actions)
                elif(elapsed > 3 and elapsed < 5.9):
                    self.arm_off=[.25, -.95, 2.1, -.8,.15, -3.25,   0,0,0.15,0,0,0 ]
                    actions.joint_positions[0:12] =  (actions.joint_positions[0:12] * 0) + self.arm_off
                    self._articulation_controller.apply_action(actions)
                elif(elapsed > 6 and elapsed < 8.99):
                    self.arm_off=[.25, -.95, 2.1, -.8,.15, -3.25,   0,0,0.38,0.15,0,0 ]
                    actions.joint_positions[0:12] =  (actions.joint_positions[0:12] * 0) + self.arm_off
                    self._articulation_controller.apply_action(actions)
                elif(elapsed > 9 and elapsed < 11.9):
                    self.arm_off=[.25, -1.05, 2.1, -.8,.15, -3.25,   0,0,0.38,0.15,0,0 ]
                    actions.joint_positions[0:12] =  (actions.joint_positions[0:12] * 0) + self.arm_off
                    self._articulation_controller.apply_action(actions)
                elif (elapsed > 12):
                    self.z = False
                    

        if self.handmode=='o' and self.pose_flag: #IK, end_effort
            self.pose_flag=False
            self.ik_conti=True
            self.ee_pose_t_p = ee_pose_w[0].copy()
            self.ee_pose_t_q = ee_pose_w[1].copy()
            self.ee_pose_t_p += self.hand_off
            print('xxx hand_off ', self.hand_off)
            print('xxx eepose target ', self.ee_pose_t_p)
            if self.websocket:
                print('xxx websocket send ')
                asyncio.run(self.send_nats_message(self.ee_pose_t_p, quat_target = self.ee_pose_t_q))
            actions_ik = self._controller.forward(target_end_effector_position=self.ee_pose_t_p, target_end_effector_orientation=self.ee_pose_t_q )
            self._articulation_controller.apply_action(actions_ik)
        if self.handmode=='o' and self.ik_conti:
           #IK, ongoing, end_effort
            print('xxx eepose curr/target conti ', ee_pose_w[0], self.ee_pose_t_p)
            print(' pose diff ', np.linalg.norm(ee_pose_w[0]-self.ee_pose_t_p))
            actions_ik = self._controller.forward(target_end_effector_position=self.ee_pose_t_p, target_end_effector_orientation=self.ee_pose_t_q )
            self._articulation_controller.apply_action(actions_ik)
            if np.linalg.norm(ee_pose_w[0]-self.ee_pose_t_p)<0.02:
                print(' pose diff ', np.linalg.norm(ee_pose_w[0]-self.ee_pose_t_p))
                self.ik_conti=False
                print("IK complete!")

        return

    async def _on_stacking_event_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_stacking_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        return

    def world_cleanup(self):
        return
