from my_sim import Simulator
from controllers.my_kin import Controller, tripod_gait
import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# Lists to store time and stability values
time_steps = []
static_stability_results = []
zmp_stability_results = []
com_x_positions = []
com_y_positions = []
reaction_forces = []  # Sum of contact forces on all legs
avg_forces = []

terrain_labels = []  

roll_values = []
pitch_values = []
yaw_values = []

controller = Controller(tripod_gait, body_height=0.15, velocity=0.46, crab_angle=-1.57)
my_sim = Simulator(controller, follow=True, visualiser=True, collision_fatal=False)

#body_height=0.15

# create steps
step_height = 0.05
step_depth = 0.2
num_steps_up = 5
num_steps_down = 5

# uneven terrain parameters
num_obstacles = 10  # Number of random obstacles
terrain_length = 2.0   # Length of uneven terrain 
terrain_width = 0.5  # Width of uneven terrain
max_bump_height = 0.025  # Maximum height variation

external_object_ids = []

frame_count = 0

# ascending stairs
for i in range(num_steps_up):
    obj_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[1.0 + i * step_depth, 0, i * step_height / 2]
    )
    external_object_ids.append(obj_id)  # store the ID
# descending stairs
for i in range(num_steps_down):
    obj_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.5, step_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[1 + (num_steps_up + i) * step_depth, 0, (num_steps_up * step_height / 2)- (i * step_height / 2) ]
    )
    external_object_ids.append(obj_id)  # store the ID
for _ in range(num_obstacles):
    x_pos = np.random.uniform(0.2 + (num_steps_up + num_steps_down) * step_depth, 
                              0.2+ (num_steps_up + num_steps_down) * step_depth + terrain_length)  
    y_pos = np.random.uniform(-terrain_width / 2, terrain_width / 2)  
    bump_height = np.random.uniform(0, max_bump_height)  
    
    obj_id = p.createMultiBody(
        baseMass=0,  
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, bump_height / 2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, bump_height / 2], rgbaColor=[0.3, 0.3, 0.3, 1]),
        basePosition=[x_pos, y_pos, bump_height / 2]  
    )
    external_object_ids.append(obj_id)  # store the ID
# run indefinitely

while True:
    my_sim.step()

    stability = my_sim.static_stability_analysis()  # Call the stability analysis
    zmp_stability = my_sim.zmp_stability_analysis()

    # Get CoM position
    com_pos, _ = my_sim.client.getBasePositionAndOrientation(my_sim.hexId)

    # Get orientation and convert to roll/pitch/yaw
    _, orientation_quat = my_sim.client.getBasePositionAndOrientation(my_sim.hexId)
    rpy = R.from_quat(orientation_quat).as_euler('xyz', degrees=True)
    roll, pitch, yaw = rpy  

    # Get total ground reaction force
    total_force = 0

    # Get total ground reaction force
    for obj_id in external_object_ids + [my_sim.groundId]:  # include both ground and other objects
        contact_points = my_sim.client.getContactPoints(my_sim.hexId, obj_id)
        total_force += sum(cp[9] for cp in contact_points)  # cp[9] = normal force at contact

        support_mask = my_sim.supporting_legs()
        num_supporting_legs = np.count_nonzero(support_mask)
        avg_force = total_force / num_supporting_legs if num_supporting_legs > 0 else 0


    #print(f"Dynamic Stability: {zmp_stability}")
    #print(f"Static Stability: {stability}")

    # Store results every N frames (to avoid excessive data points)
    if frame_count % 10 == 0:
        time_steps.append(frame_count)
        static_stability_results.append(stability)
        zmp_stability_results.append(zmp_stability)
        com_x_positions.append(com_pos[0])
        com_y_positions.append(com_pos[1])
        reaction_forces.append(total_force)
        avg_forces.append(avg_force)
        roll_values.append(roll)
        pitch_values.append(pitch)
        yaw_values.append(yaw)

        x = com_pos[0]
        if x < 1:
            terrain_labels.append("Start")
        elif x < 1 + num_steps_up * step_depth:
            terrain_labels.append("Ascending")
        elif x < 1 + (num_steps_up + num_steps_down) * step_depth:
            terrain_labels.append("Descending")
        else:
            terrain_labels.append("Flat")

    
    frame_count += 1
    
    # Exit condition (define your own)
    if frame_count > 3000:  # Example: Stop after 1000 frames
        break

def shade_terrain(ax, time_steps, terrain_labels):
    current_label = terrain_labels[0]
    start = time_steps[0]

    for i in range(1, len(time_steps)):
        if terrain_labels[i] != current_label or i == len(time_steps) - 1:
            end = time_steps[i]
            color = {
                "Ascending": "#95C7FF",
                "Descending": "#FF8989",
                "Flat": "#87FF87",
                "Start": "#f0f0f0"
            }.get(current_label, "#ffffff")

            ax.axvspan(start, end, facecolor=color, alpha=0.3, label=current_label if start == time_steps[0] else "")
            current_label = terrain_labels[i]
            start = end


# Plot Stability Analysis Over Time
plt.figure(figsize=(12, 6))

ax1 = plt.subplot(2, 2, 1)
plt.plot(time_steps, static_stability_results, label="Static Stability", linestyle = 'dotted')
plt.plot(time_steps, zmp_stability_results, label="ZMP Stability", linestyle = 'dashed')
shade_terrain(ax1, time_steps, terrain_labels)
plt.xlabel("Time Steps")
plt.ylabel("Stability (1 = Stable, 0 = Unstable)")
plt.title("Hexapod Stability Over Time")
plt.legend()
plt.grid()

# Plot Center of Mass (CoM) Position
ax2 = plt.subplot(2, 2, 2)
plt.plot(time_steps, com_x_positions, label="CoM X", linestyle = 'dotted')
plt.plot(time_steps, com_y_positions, label="CoM Y", linestyle = 'dashed')
shade_terrain(ax2, time_steps, terrain_labels)
plt.xlabel("Time Steps")
plt.ylabel("CoM Position (m)")
plt.title("Center of Mass Position Over Time")
plt.legend()
plt.grid()

# Plot Ground Reaction Force
ax3 = plt.subplot(2, 2, 3)
#plt.plot(time_steps, reaction_forces, label="Total Ground Reaction Force", linestyle = 'dotted', color="red")
plt.plot(time_steps, avg_forces, label="Avg Ground Reaction Force", linestyle = 'dotted', color="blue")
shade_terrain(ax3, time_steps, terrain_labels)
plt.xlabel("Time Steps")
plt.ylabel("Force (N)")
plt.title("Ground Reaction Force Over Time")
plt.legend()
plt.grid()

# Plot Tilt (Roll and Pitch)
ax4 = plt.subplot(2, 2, 4)
plt.plot(time_steps, roll_values, label="Roll (deg)", linestyle='dashed', color="purple")
plt.plot(time_steps, pitch_values, label="Pitch (deg)", linestyle='dashed', color="orange")
shade_terrain(ax4, time_steps, terrain_labels)
#plt.plot(time_steps, yaw_values, label="Yaw (deg)", linestyle='dashed', color="green")
plt.xlabel("Time Steps")
plt.ylabel("Angle (deg)")
plt.title("Tilt (Roll & Pitch) Over Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()








     


