import pybullet as p
import numpy as np
from scipy.spatial import ConvexHull
from my_sim import Simulator


def is_stable(robotId, contact_legs): 
        # robotId (int): ID of the hexapod body in PyBullet.
        # leg_ids (list): List of leg IDs in PyBullet.
        # returns (bool, float): (True if stable, stability margin)
        
        contact_points = []
        
        # Get contact points for each leg
        for leg in contact_legs:
            contact = p.getContactPoints(bodyA=robotId, linkIndexA=leg)
            if contact:  # If the leg is in contact with the ground
                contact_points.append(contact[0][6])  # Contact point position (x, y)
        
        # If less than 3 legs are in contact, the robot is unstable
        if len(set(contact_points)) < 3:  # Use `set()` to avoid duplicate contacts
            print(f"Detected contact legs: {contact_points}")
            print("Warning: Less than 3 legs in contact!")
        return False, -1



        # Convert to NumPy array for processing
        contact_points = np.array(contact_points)

        # Compute Convex Hull (Support Polygon)
        hull = ConvexHull(contact_points[:, :2])  # Only consider x, y for 2D polygon

        # Compute Center of Mass (CoM)
        com_position, _ = p.getBasePositionAndOrientation(robotId)
        com_x, com_y = com_position[:2]  # Only x, y for stability check

        # Check if CoM is inside the support polygon
        hull_path = contact_points[hull.vertices]  # Get the vertices of the convex hull
        hull_path = np.vstack([hull_path, hull_path[0]])  # Close the polygon

        # Stability check using Shoelace Algorithm
        inside = is_point_inside_polygon(com_x, com_y, hull_path)

        # Compute Stability Margin (Distance from CoM to nearest support polygon edge)
        margin = stability_margin(com_x, com_y, hull_path) if inside else 0

        return inside, margin
'''
def is_point_inside_polygon(x, y, polygon):
        """
        Check if a point (x, y) is inside a polygon using the winding number algorithm.
        """
        crossings = 0
        for i in range(len(polygon) - 1):
            x1, y1, _ = polygon[i]  # Ignore the third value (z)
            x2, y2, _ = polygon[i + 1]

            if ((y1 <= y < y2) or (y2 <= y < y1)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                crossings += 1

        return crossings % 2 == 1  # Odd number of crossings → inside

def stability_margin(com_x, com_y, polygon):
        """
        Compute the stability margin as the shortest distance from CoM to the support polygon edges.
        """
        min_distance = float("inf")

        for i in range(len(polygon) - 1):
            x1, y1 = polygon[i]
            x2, y2 = polygon[i + 1]

            # Compute perpendicular distance from CoM to line segment
            A, B, C = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
            distance = abs(A * com_x + B * com_y + C) / np.sqrt(A**2 + B**2)

            min_distance = min(min_distance, distance)

        return min_distance
'''