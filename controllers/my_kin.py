import numpy as np

# radius, offset, step_height, phase, duty_factor
tripod_gait = [	0.15, 0, 0.05, 0.5, 0.5, # leg 1
				0.15, 0, 0.05, 0.0, 0.5, # leg 2
				0.15, 0, 0.05, 0.5, 0.5, # leg 3
				0.15, 0, 0.05, 0.0, 0.5, # leg 4
				0.15, 0, 0.05, 0.5, 0.5, # leg 5
				0.15, 0, 0.05, 0.0, 0.5] # leg 6

stationary = [0.18, 0, 0, 0, 0] * 6

#Radius: Determines how far the leg moves outward from the center
#Offset: Shifts the step cycle relative to the body
#Step Height: The maximum vertical movement of the foot
#Phase: Determines when the leg is lifted within the gait cycle
#Duty Factor: The fraction of time a leg spends on the ground

class Controller:
	def __init__(self, params=tripod_gait, crab_angle=0.0, body_height=0.14, period=1.0, velocity=0.1, dt=1/240):
		#params= tripod_gait
		#crab_angle=0.0: Controls the walking direction. A value of 0° means forward walking
		#body_height=0.14: The default height of the robot's body from the ground
		#period=1.0: The time (in seconds) taken to complete one full gait cycle
		#velocity=0.1: The forward walking speed
		#dt=1/240: The time step (smallest unit of time in simulation)
                
		self.l_1 = 0.05317 # link lengths of each leg
		self.l_2 = 0.10188
		self.l_3 = 0.14735

		self.dt = dt # gait Control Variables
		self.period = period
		self.velocity = velocity
		self.crab_angle = crab_angle
		self.body_height = body_height

		self.array_dim = int(np.around(period / dt)) # number of simulation steps per gait cycle

		self.positions = np.empty((0, self.array_dim)) # initialize empty arrays for storing
		self.velocities = np.empty((0, self.array_dim))

		self.angles = np.empty((0, self.array_dim))
		self.speeds = np.empty((0, self.array_dim))

		params = np.array(params).reshape(6, 5) # converts params into a 6×5 matrix
		# 6 represents the six legs of the hexapod
		# 5 represents different phases or parameters for the gait

		for leg_index in range(6): # generates and verifies leg trajectories
			foot_positions, foot_velocities = self.__leg_traj(leg_index, params[leg_index]) # calculates the foot trajectory (positions and velocities) for each leg based on its gait parameters

			joint_angles, joint_speeds = self.__inverse_kinematics(foot_positions, foot_velocities) # translates foot positions into joint angles

			achieved_positions = self.forward_kinematics(joint_angles) # verify if the computed joint angles result in the expected foot positions
			valid = np.all(np.isclose(foot_positions, achieved_positions))
			if not valid:
				raise RuntimeError('Desired foot trajectory not achieveable')

			self.positions = np.append(self.positions, foot_positions, axis=0) # stores the Computed Values
			self.velocities = np.append(self.velocities, foot_velocities, axis=0)
			self.angles = np.append(self.angles, joint_angles, axis=0)
			self.speeds = np.append(self.speeds, joint_speeds, axis=0)


	def joint_angles(self, t): # retrieves joint angles at a specific time t
		k = int(((t % self.period) / self.period) * self.array_dim)
		return self.angles[:, k] # extracts the joint angles for all six legs at the computed index k
		# returns a 6×3 matrix (each row = joint angles of one leg)

	def joint_speeds(self, t): # retrieves the joint speeds at a given time t
		k = int(((t % self.period) / self.period) * self.array_dim)
		return self.speeds[:, k] # extracts the joint speeds for all six legs at the computed index k
		# returns a 6×3 matrix (each row = joint speeds of one leg)

	def __leg_traj(self, leg_index, leg_params): # returns x, y, z trajectory path points for a leg in the leg coordinate space
		leg_angle = (np.pi / 3.0) * (leg_index) # each leg is 60° apart
		radius, offset, step_height, phase, duty_factor = leg_params # extract Gait Parameters
		stride = self.velocity * duty_factor * self.period # distance the leg moves in one step

		mid = np.zeros(3) # midpoint (Highest Step Position)
		mid[0] = radius * np.cos(offset)  
		mid[1] = radius * np.sin(offset)
		mid[2] = -self.body_height + 0.014 + step_height

		start = np.zeros(3) # start of Step (Ground Level)
		start[0] = mid[0] + (stride / 2) * np.cos(-leg_angle + self.crab_angle)
		start[1] = mid[1] + (stride / 2) * np.sin(-leg_angle + self.crab_angle)
		start[2] = -self.body_height + 0.014
		# moves x, y forward

		end = np.zeros(3) # end of Step (Ground Level)
		end[0] = mid[0] - (stride / 2) * np.cos(-leg_angle + self.crab_angle)
		end[1] = mid[1] - (stride / 2) * np.sin(-leg_angle + self.crab_angle)
		end[2] = -self.body_height + 0.014
		# moves x, y backward

		# compute support path
		support_dim = int(np.around(self.array_dim * duty_factor)) # support phase lasts for a fraction of the total gait cycle, determined by duty_factor
		support_positions, support_velocities = self.__support_traj(start, end, support_dim) # calculates foot positions & velocities while it's in contact with the ground

		# compute swing path
		swing_dim = int(np.around(self.array_dim * (1.0 - duty_factor))) # swing phase is the remaining portion of the cycle (1.0 - duty_factor)
		swing_positions, swing_velocities = self.__swing_traj(end, mid, start, swing_dim) # computes the foot’s arc-like trajectory

		positions = np.append(support_positions, swing_positions, axis=1) # merges both phases into a single continuous trajectory
		velocities = np.append(support_velocities, swing_velocities, axis=1)

		phase_shift = int(np.around(phase * self.array_dim)) # shift points according to phase
		positions = np.roll(positions, phase_shift, axis=1) # shifts the motion pattern to align with the specified gait phase
		velocities = np.roll(velocities, phase_shift, axis=1) 

		return positions, velocities
	
	def __support_traj(self, start, end, num): # computes the trajectory for the support phase
		
		positions = np.linspace(start, end, num, axis=1) # generate num points between start and end

		duration = num * self.dt # calculates total time for the support phase

		with np.errstate(divide='ignore', invalid='ignore'): # avoid division errors
			velocity = ((end - start) / duration).reshape(3,1) # compute Velocity

		velocities = np.tile(velocity, num) # repeats the velocity vector to match the number of time steps to ensure constant velocity

		return positions, velocities
	
	def __swing_traj(self, start, via, end, num): # computes the trajectory for the swing phase
		t = np.ones((7, num)) # num is the number of points in the swing trajectory
		tf = num * self.dt # total time for the swing phase
		time = np.linspace(0, tf, num) # creates evenly spaced time points

		for i in range(7): # creates a 7-row matrix where each row is time^i (for i = 0 to 6)
			t[i,:] = np.power(time, i)

		a_0 = start # starting point
		a_1 = np.zeros(3) # set to zero to ensure initial velocity is zero
		a_2 = np.zeros(3) # set to zero to ensure initial acceleration is zero

		a_3 = (2 / (tf ** 3)) * (32 * (via - start) - 11 * (end - start))
		a_4 = -(3 / (tf ** 4)) * (64 * (via - start) - 27 * (end - start))
		a_5 = (3 / (tf ** 5)) * (64 * (via - start) - 30 * (end - start))
		a_6 = -(32 / (tf ** 6)) * (2 * (via - start) - (end - start))
		#coefficients control how the foot moves through space with a smooth curve

		# constructs a smooth trajectory for the foot's swing phase using a 6th-degree polynomial
		# P(t) = a0 + a1t + a2t^2 + a3t^3 + a4t^4 + a5t^5 + a6t^6 

		positions = np.stack([a_0, a_1, a_2, a_3, a_4, a_5, a_6], axis=-1).dot(t) # positions are computed for each foot position (x, y, z)
		velocities = np.stack([a_1, 2*a_2, 3*a_3, 4*a_4, 5*a_5, 6*a_6, np.zeros(3)], axis=-1).dot(t) # velocities are computed as the derivative of the position equation

		return positions, velocities
	
	def __inverse_kinematics(self, foot_position, foot_speed): # computes joint angles and speeds needed for a hexapod leg to reach a desired foot position and velocity
		l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3 # link lengths

		x, y, z = foot_position
		dx, dy, dz = foot_speed

		#joint_angles (theta_1, theta_2, theta_3): required joint angles
		#joint_speeds (theta_dot_1, theta_dot_2, theta_dot_3): required joint velocities

		theta_1 = np.arctan2(y, x) #coxa rotation

		c_1, s_1 = np.cos(theta_1), np.sin(theta_1)
		c_3 = ((x - l_1 * c_1)**2 + (y - l_1 * s_1)**2 + z**2 - l_2**2 - l_3**2) / (2 * l_2 * l_3)
		s_3 = -np.sqrt(np.maximum(1 - c_3**2, 0)) # maximum ensures not negative

		theta_2 = np.arctan2(z, (np.sqrt((x - l_1 * c_1)**2 + (y - l_1 * s_1)**2))) - np.arctan2((l_3 * s_3), (l_2 + l_3 * c_3)) # femur rotation
		theta_3 = np.arctan2(s_3, c_3) # tibia rotation

		c_2, s_2 = np.cos(theta_2), np.sin(theta_2)
		c_23 = np.cos(theta_2 + theta_3)

		# compute Joint Velocities
		with np.errstate(all='ignore'):
			theta_dot_1 = (dy*c_1 - dx*s_1) / (l_1 + l_3*c_23 + l_2*c_2)
			theta_dot_2 = (1/l_2)*(dz*c_2 - dx*c_1*s_2 - dy*s_1*s_2 + (c_3 / s_3)*(dz*s_2 + dx*c_1*c_2 + dy*c_2*s_1))
			theta_dot_3 = -(1/l_2)*(dz*c_2 - dx*c_1*s_2 - dy*s_1*s_2 + ((l_2 + l_3*c_3)/(l_3*s_3))*(dz*s_2 + dx*c_1*c_2 + dy*c_2*s_1))

		# ensures the calculations don’t produce NaNs or errors
		theta_dot_1 = np.nan_to_num(theta_dot_1, nan=0.0, posinf=0.0, neginf=0.0)
		theta_dot_2 = np.nan_to_num(theta_dot_2, nan=0.0, posinf=0.0, neginf=0.0)
		theta_dot_3 = np.nan_to_num(theta_dot_3, nan=0.0, posinf=0.0, neginf=0.0)

		joint_angles = np.array([theta_1, theta_2, theta_3])
		joint_speeds = np.array([theta_dot_1, theta_dot_2, theta_dot_3])

		return joint_angles, joint_speeds
	
	def forward_kinematics(self, joint_angles): # computes the foot position (x, y, z) from given joint angles (theta_1, theta_2, theta_3)
		l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3
		theta_1, theta_2, theta_3 = joint_angles
		
		x = np.cos(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
		y = np.sin(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
		z = l_3 * np.sin(theta_2 + theta_3) + l_2 * np.sin(theta_2)

		return np.array([x, y, z])

	def IMU_feedback(self, measured_attitude):
		return

def reshape(x): # converts and scales an input array x into Body Height, Velocity, Leg Gait Parameters 
		
		x = np.array(x) #convert Input to NumPy Array
	
		# extract Body Height and Velocity
		height = x[0] * 0.2
		velocity = x[1] * 0.5

		leg_params = x[2:].reshape((6,5))  # reshaped into a (6,5) matrix:  6 legs, each with 5 gait parameters(radius, offset, step_height, phase, duty_cycle)

		param_min = np.array([0.0, -1.745, 0.01, 0.0, 0.0]) # min allowed values for each parameter
		param_max = np.array([0.3, 1.745, 0.2, 1.0, 1.0]) # max allowed values for each parameter
		
		leg_params = leg_params * (param_max - param_min) + param_min # scales ans shifts each parameter to its proper range

		return height, velocity, leg_params

if __name__ == '__main__':
	import time

	# radius, offset, step, phase, duty_cycle
	leg_params = [
	0.1, 0, 0.1, 0.0, 0.5, # leg 0
	0.1, 0, 0.1, 0.5, 0.5, # leg 1
	0.1, 0, 0.1, 0.0, 0.5, # leg 2
	0.1, 0, 0.1, 0.5, 0.5, # leg 3
	0.1, 0, 0.1, 0.0, 0.5, # leg 4
	0.1, 0, 0.1, 0.5, 0.5] # leg 5

	start = time.perf_counter()
	ctrl = Controller(leg_params) # initializes Controller
	end = time.perf_counter()

	print((end-start)*1000)
	