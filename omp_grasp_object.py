# Initialize SimulationAPP
from omni.isaac.kit import SimulationApp
config = {
    'width': 1920,
    'height': 1080,
    'headless': False,
    'window_width': 2560,
    'window_height': 1440,
}

simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
import omni.usd
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.rotations import euler_angles_to_quat

import sys, os
import numpy as np

curr_file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_file_dir)

from robots.omp.omp_system import OmpSystem
from robots.omp.omp_rmpflow_controller import RMPFlowController
from robots.omp.omp_manipulation_controller import OmpManipulationController

my_world = World(stage_units_in_meters=1.0)

# Initialize the Scene
scene = my_world.scene
scene.add_default_ground_plane()

cube = DynamicCuboid(prim_path = "/World/cube",  name = "cube",
                    position = [0.2, -0.3, 0.0],
                    color = np.array([1, 0.25, 0]), size = 0.05, mass = 0.01) 
scene.add(cube)

obstacle = VisualCuboid(
            prim_path="/World/obstacle",
            position= [0.1, -0.3, 0.0],
            size=1.0,
            scale=np.array([0.05, 0.3, 0.5]),
            color=np.array([0, 0, 1.0]),
        )
scene.add(obstacle)

my_robot = OmpSystem(
    prim_path="/World/omp", # should be unique
    name="uid_omp", # should be unique, used to access the object 
    usd_path=os.path.join(curr_file_dir, "robots/omp/omp_rhp12rn.usd"),
    #activate_camera=False,
    )

scene.add(my_robot)

stage = omni.usd.get_context().get_stage()
XformOmp = UsdGeom.Xformable(stage.GetPrimAtPath("/World/omp"))
TransfOmp = XformOmp.ComputeLocalToWorldTransform(0)  
invTransfOmp = TransfOmp.GetInverse()
print(invTransfOmp)

my_controller = OmpManipulationController(
    name='omp_manipulation_controller',
    # 로봇 모션 controller 설정
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_robot
    ),
    gripper=my_robot.gripper,
    # phase의 진행 속도 설정
    events_dt=[0.008],
)

my_controller.addObstacle(obstacle=obstacle)

# # robot control(PD control)을 위한 instance 선언
articulation_controller = my_robot.get_articulation_controller()

my_controller.reset()

# Simulation Loop
my_world.reset()

init_target_position = my_robot._end_effector.get_world_pose()[0]

state = "APPROACH0"
waitCount = 0

#gripper offset & orientation
end_effector_offset = np.array([0, 0, 0.12])
end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))

while simulation_app.is_running():
    
    my_world.step(render=True)
    
    robot_pos, robot_ori = my_robot.get_world_pose()
    my_controller.rmpflow.set_robot_base_pose(robot_pos, robot_ori, False)
    
    #print(my_robot.get_joints_state().positions)

    if my_world.is_playing():

        if state == "APPROACH0":
            # get cube position 
            cube_position = cube.get_world_pose()[0]
            cube_position[2] = 0.2

            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.forward(
                target_position=cube_position,
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset,
                end_effector_orientation=end_effector_orientation
                )
            
            # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done position control of end-effector")
                my_controller.reset()

                waitCount += 1
                if waitCount > 3:    
                    state = "APPROACH1"
                    waitCount = 0

        # APPROACH 하는 state에 대한 action 수행
        if state == "APPROACH1":
            # cube 위치 얻어오기
            cube_position = cube.get_world_pose()[0]


            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.forward(
                target_position=cube_position,
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset,
                end_effector_orientation=end_effector_orientation
                )
            
            # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done position control of end-effector")
                my_controller.reset()

                waitCount += 1
                if waitCount > 2:    
                    state = "GRASP"
                    waitCount = 0
        
        elif state == "GRASP":
            actions = my_controller.close(
                 current_joint_positions=my_robot.get_joints_state().positions,
                 end_effector_offset = end_effector_offset)
            
            if my_controller.is_done():
                print("done grasping")
                my_controller.reset()
                # GRASP가 끝났을 경우 LIFT state 단계로 변경
                 
                state = "LIFT"
        
        elif state == "LIFT":
            actions = my_controller.forward(
                target_position=init_target_position,
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset,
                end_effector_orientation=end_effector_orientation
            )

            if my_controller.is_done():
                print("done lifting")
                my_controller.reset()
                # LIFT가 끝났을 경우 OPEN state 단계로 변경
                state = "OPEN"


        elif state == "OPEN":
            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.open(
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset,
            )

            # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done lifting")
                my_controller.reset()
                # LIFT가 끝났을 경우 APPROACH state 단계로 변경
                state = "APPROACH0"

    articulation_controller.apply_action(actions)

simulation_app.close()