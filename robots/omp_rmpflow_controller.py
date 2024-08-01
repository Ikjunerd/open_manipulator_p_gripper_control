# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation
from pxr import Gf, UsdGeom
import os
import numpy as np

curr_file_dir = os.path.dirname(os.path.realpath(__file__))

class RMPFlowController(mg.MotionPolicyController):
    """[summary]

        Args:
            name (str): [description]
            robot_articulation (Articulation): [description]
            physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
            attach_gripper (bool, optional): [description]. Defaults to False.
        """

    def __init__(
        self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0, attach_gripper: bool = False
    ) -> None:

        # if attach_gripper:
        #     self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
        #         "UR5e", "RMPflow")
        #else:  #self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
                
        self.rmp_flow = mg.lula.motion_policies.RmpFlow( 
            robot_description_path = os.path.join(curr_file_dir, "rmpflow_config\\omp_robot_descriptor.yaml"),
            urdf_path = os.path.join(curr_file_dir, "rmpflow_config\\omp.urdf"),
            rmpflow_config_path = os.path.join(curr_file_dir, "rmpflow_config\\omp_rmpflow_config.yaml"),
            end_effector_frame_name = "end_link",            
            maximum_substep_size = 0.00334)
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )        
        print(self._default_position, self._default_orientation)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
       
    def set_robot_base_pose(self, position, orientation, visualize_collision_spheres):
        self._default_position = position
        self._default_orientation = orientation
        self.rmp_flow.set_robot_base_pose(robot_position=self._default_position, robot_orientation=self._default_orientation)
        if visualize_collision_spheres: 
            self.rmp_flow.visualize_collision_spheres()

    def set_obstacle(self, obstacle):
        self.rmp_flow.add_obstacle(obstacle)