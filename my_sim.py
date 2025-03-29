import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from scipy.spatial import ConvexHull

class Simulator:
    def __init__(self, controller, urdf='/urdf/hexapod_simplified.urdf', visualiser=False, follow=True, collision_fatal=True, camera_position=[0, 0, 0], camera_distance=0.7, camera_yaw=20, camera_pitch=-30): #initialize values
        self.t = 0  # Current simulation time
        self.dt = 1.0 / 240.0  # PyBullet default timestep (1/240s)
        self.n_step = 0  # Step counter
        self.gravity = -9.81  # Gravity (downward in z-axis)
        self.foot_friction = 0.7  # Friction coefficient of the feet
        self.controller = controller
        self.visualiser_enabled = visualiser
        self.follow = follow
        self.collision_fatal = collision_fatal

        # camera settings
        self.camera_position = [0.7, 0, 0] 
        self.camera_distance = 0.8 
        self.camera_yaw = 0 
        self.camera_pitch = -45

        self.camera_position = camera_position
        self.camera_distance = camera_distance # distance from the focus point
        self.camera_yaw = camera_yaw # horizontal rotation
        self.camera_pitch = camera_pitch # vertical tilt

        connection_mode = p.GUI if self.visualiser_enabled else p.DIRECT # Runs PyBullet in GUI mode 
        self.client = bc.BulletClient(connection_mode=connection_mode)

        if self.visualiser_enabled:
            self.client.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=self.camera_position)
            self.client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

        self.client.setAdditionalSearchPath(pybullet_data.getDataPath()) # default search path for PyBullet assets like URDFs
        self.client.setGravity(0, 0, self.gravity)
        self.client.setRealTimeSimulation(False)  # simulation needs to be explicitly stepped

        # loads ground plane
        self.groundId = self.client.loadURDF('plane.urdf') 
        self.client.changeDynamics(self.groundId, -1, lateralFriction=self.foot_friction)


		# adds hexapod URDF 
        position = [0, 0, self.controller.body_height]
        orientation = self.client.getQuaternionFromEuler([0, 0, -controller.crab_angle])
        filepath = os.path.abspath(os.path.dirname(__file__)) + urdf # directory path
        self.hexId = self.client.loadURDF(filepath, position, orientation, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION) 
        
		# get joint and link info from model
        self.joints = self.__get_joints(self.hexId) # List of joint indeces
        self.links = self.__get_links(self.joints) # List of link indeces

		# initialise joints and links
        self.__init_joints(self.controller, self.joints)
        self.__init_links(self.links)

	# set joints to their initial positions
    def __init_joints(self, controller, joints):
        joint_angles = controller.joint_angles(t=0)

        for index, joint in enumerate(joints):
            joint_angle = joint_angles[index]
            joint_index, lower_limit, upper_limit, max_torque, max_speed = joint # joint ID, min & max joint angles, actuator limits
			
            if joint_index is None: continue # skipping Non-Existent Joints
                  
			# set joints to their starting position
            self.client.resetJointState(self.hexId, joint_index, targetValue=joint_angle)

			# assign small friction force to joint to simulate servo friction
            self.client.setJointMotorControl2(self.hexId, joint_index, p.VELOCITY_CONTROL, force=0.1)

    def __init_links(self, links):
        tibia_links = links[:, 2] # Select the tibia links
        for link_index in tibia_links:
            self.client.changeDynamics(self.hexId, link_index, lateralFriction=self.foot_friction)

        femur_links = links[:, 1]  # Select the femur links
        for link_index in femur_links:
            # Disable Collisions Between Femur and Base
            self.client.setCollisionFilterPair(self.hexId, self.hexId, linkIndexA=-1, linkIndexB=link_index, enableCollision=0)

    def __get_joints(self, robotId): # fetches and stores the joint index and joint information
        joint_names = [b'joint_1_1', b'joint_1_2', b'joint_1_3', 
                       b'joint_2_1', b'joint_2_2', b'joint_2_3',
                       b'joint_3_1', b'joint_3_2', b'joint_3_3',
                       b'joint_4_1', b'joint_4_2', b'joint_4_3',
                       b'joint_5_1', b'joint_5_2', b'joint_5_3',
                       b'joint_6_1', b'joint_6_2', b'joint_6_3'] # in byte strings, as required by PyBullet
        
        joints = np.full((len(joint_names), 5), None) # Creates an empty NumPy array with shape (18, 5)
        # Each row represents a joint
        # Each column will store joint information

        for joint_index in range(self.client.getNumJoints(robotId)): 
            info = self.client.getJointInfo(robotId, joint_index) # Iterates through all joints and retrieves joint data

            try: 
                index = joint_names.index(info[1])
				# [ joint_index, lower_limit, upper_limit, max_torque, max_velocity ]
                joints[index] = [info[0], info[8], info[9], info[10], info[11]]
            except ValueError:
                print('Unexpected joint name in URDF')
        return joints

    def __get_links(self, joints): # extracts the link indices from the given joint information and organizes them into a 6×3 matrix
        link_indices = self.joints[:,0]
        links = link_indices.reshape(6,3)
        #[[coxa_1, femur_1, tibia_1],
        #[coxa_2, femur_2, tibia_2],
        #[coxa_3, femur_3, tibia_3],
        #[coxa_4, femur_4, tibia_4],
        #[coxa_5, femur_5, tibia_5],
        #[coxa_6, femur_6, tibia_6]]

        #return joints[:, 0].reshape(6, 3)  # Directly reshape without storing in an intermediate variable

        return links

    def set_foot_friction(self, foot_friction): # updates the lateral friction of the ground and stores the new value.
        self.foot_friction = foot_friction
        self.client.changeDynamics(self.groundId, -1, lateralFriction=foot_friction)

    def terminate(self): # prints PyBullet error if termination failed
        try:
            self.client.disconnect()  # Disconnects PyBullet simulation
        except p.error as e:
            print('Termination of simulation failed:', e) # prints error if disconnection fails

    def step(self): # function runs at each simulation step and ensures the hexapod moves correctly

        start_time = time.perf_counter()
        joint_angles = self.controller.joint_angles(t=self.t)

        for index, joint_properties in enumerate(self.joints): # iterates over all joints
            joint_index, lower_limit, upper_limit, max_torque, max_speed = joint_properties
			
            if (joint_index is None): continue # skips if joint is not present 

            joint_angle = joint_angles[index] # fetches the computed target joint angle
			
            joint_angle = min(max(lower_limit, joint_angle), upper_limit) # clamps joint angles within limits

            self.client.setJointMotorControl2(self.hexId, joint_index, p.POSITION_CONTROL, targetPosition=joint_angle, force=max_torque, maxVelocity=max_speed) # sets Joint Motor Control
            #p.POSITION_CONTROL: Moves the joint to targetPosition like a servo
            #targetPosition=joint_angle : The computed angle for this step
            #force=max_torque : Limits the applied torque
            #maxVelocity=max_speed : Limits the movement speed
               
        if self.collision_fatal: # Detects self-collisions or ground collisions
            if self.__link_collision() or self.__ground_collision():
                raise RuntimeError('Link collision during simulation') #Raises an error if collision occurs

        end_time = time.perf_counter()

        if self.visualiser_enabled: #Moves the camera to track the hexapod
            if self.follow:
                self.client.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=self.base_pos())
                    
            time.sleep(max(self.dt - end_time + start_time, 0)) # Ensures a fixed time step

        if ((self.n_step % 24) == 0): # Sends IMU feedback every 24 steps (~10Hz update rate) to allow realistic sensor input
            self.controller.IMU_feedback(self.base_orientation())

        self.client.stepSimulation() # Increments simulation step count
        self.n_step += 1
        self.t += self.dt
	
    def supporting_legs(self): # determines which legs of the hexapod are currently in contact with the ground
		
        tibia_links = self.links[:, 2] #extracts the tibia links of each leg
          
        contact_points = np.array(self.client.getContactPoints(self.hexId, self.groundId), dtype=object) # returns a list of all contact points between the hexapod and ground

        try:
            contact_links = contact_points[:, 3] # extracts the link indices in contact with the ground
        except IndexError as e: 
            contact_links = np.array([])
               
        supporting_legs = np.isin(tibia_links, contact_links) # which legs are in contact with the ground
          #True: The tibia is touching the ground
          #False: The tibia is in the air

        return supporting_legs

    def __link_collision(self): # returns true for collision between links in robot

        contact_points = np.asarray(self.client.getContactPoints(self.hexId, self.hexId), dtype=object) # checks for collisions within the hexapod itself

        return contact_points.size > 0 # returns True if collision detected

    def __ground_collision(self): # returns true for collisions between the robot and the ground
        tibia_links = self.links[:, 2] # get the tibia links

        contact_points = np.array(self.client.getContactPoints(self.hexId, self.groundId), dtype=object) # get contact points between hexapod and ground

        try: # extract the list of robot links
            contact_links = contact_points[:, 3]
        except IndexError as e:
            contact_links = np.array([])
		
        contact_links = contact_links[~np.isin(contact_links, tibia_links)] # exclude tibia links

        return contact_links.size > 0 # return True if any non-foot part is touching the ground

    def base_orientation(self): # retrieves the current orientation of the hexapod 
        quaternion = self.client.getBasePositionAndOrientation(self.hexId)[1]
        return p.getEulerFromQuaternion(quaternion)

    def base_pos(self): # retrieves the current position of the hexapod’s base without including its orientation
        return self.client.getBasePositionAndOrientation(self.hexId)[0] # extracts only the position
    

def evaulate_gait(leg_params, body_height=0.14, velocity=0.3, duration=5.0, visualiser=True, collisions=False): # tests a hexapod gait by running a simulated walk cycle and evaluating its performance

        #leg_params: Defines the leg movement parameters
        #body_height: Initial height of the hexapod
        #velocity: Walking speed
        #duration: How long the gait test runs
        #visualiser: Enables/disables PyBullet rendering
        #collisions: Whether collisions cause failure

        controller = Controller(leg_params, body_height=body_height, velocity=velocity, crab_angle=-np.pi/6) # handles the gait logic

        simulator = Simulator(controller, follow=True, visualiser=visualiser, collision_fatal=collisions) # runs the physics simulation with the given controller

        contact_sequence = np.full((6, 0), False) # stores foot contact

        for t in np.arange(0, duration, step=simulator.dt): # iterates over the simulation time
            try:
                 simulator.step()
            except RuntimeError as error:
                print(error)
            fitness = 0
            break
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1,1), axis=1)

        fitness = simulator.base_pos()[0] # the X-position of the hexapod after the simulation (this measures forward movement)

        descriptor = np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1) # the percentage of time each leg is touching the ground

        simulator.terminate()

        return fitness, descriptor


if __name__ == "__main__":
        from controllers.my_kin import Controller
        from controllers.my_kin import stationary
	
        controller = Controller(stationary, body_height=0.11, velocity=0.0, crab_angle=-np.pi/6)
        #stationary: No movement (legs stay in place)
        #body_height=0.11: The hexapod's body is 11 cm above the ground
        #velocity=0.0: The robot does not move (static pose)
        #crab_angle=-π/6: Rotates the gait 30° to the left

        my_sim = Simulator(controller=controller, follow=False, visualiser=True, collision_fatal=False, camera_distance=1.0, camera_yaw=90, camera_pitch=-55)
        #follow=False: The camera does not track the robot
        #visualiser=True: Enables PyBullet GUI
        #collision_fatal=False: The robot won't stop if a collision happens
        #camera_distance=1.0: Zoom level of the PyBullet camera
        #camera_yaw=90, camera_pitch=-55: Camera angle (side view)

        while True:
		        my_sim.step()   

