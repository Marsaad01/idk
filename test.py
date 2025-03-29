from my_sim import Simulator
from controllers.my_kin import Controller, tripod_gait
from stab import is_stable
import pybullet as p
import pybullet_data
import numpy as np
import time

controller = Controller(tripod_gait, body_height=0.15, velocity=0.46, crab_angle=-1.57)
my_sim = Simulator(controller, follow=True, visualiser=True, collision_fatal=False)

# create steps
step_height = 0.05
step_depth = 0.2
num_steps_up = 0
num_steps_down = 0

# uneven terrain parameters
num_obstacles = 0  # Number of random obstacles
terrain_length = 5.0   # Length of uneven terrain 
terrain_width = 5.0  # Width of uneven terrain
max_bump_height = 0.05  # Maximum height variation

# ascending stairs
for i in range(num_steps_up):
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[0.4 + i * step_depth, 0, i * step_height / 2]
    )

# descending stairs
for i in range(num_steps_down):
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[0.4 + (num_steps_up + i) * step_depth, 0, (num_steps_up * step_height / 2)- (i * step_height / 2) ]
    )

for _ in range(num_obstacles):
    x_pos = np.random.uniform(0.4 + (num_steps_up + num_steps_down) * step_depth, 
                              0.4 + (num_steps_up + num_steps_down) * step_depth + terrain_length)  
    y_pos = np.random.uniform(-terrain_width / 2, terrain_width / 2)  
    bump_height = np.random.uniform(0, max_bump_height)  
    
    p.createMultiBody(
        baseMass=0,  
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, bump_height / 2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, bump_height / 2], rgbaColor=[0.3, 0.3, 0.3, 1]),
        basePosition=[x_pos, y_pos, bump_height / 2]  
    )
# Get hexapod robot ID
robotId = my_sim.hexId  # Assuming your simulator has a robot ID

# run indefinitely
while True:
    my_sim.step()

    # Get supporting legs from the simulator
    contact_legs = np.where(my_sim.supporting_legs())[0].tolist()  # Convert boolean mask to leg indices
    print(f"Supporting Legs: {contact_legs}")  # Debugging
    time.sleep(0.01)
    stable, margin = is_stable(robotId, contact_legs)
    
    if stable:
        print(f"Hexapod is stable. Stability margin: {margin:.4f}")
    else:
        print("Hexapod is unstable!")


     


