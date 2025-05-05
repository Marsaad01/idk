    def static_stability_analysis(self):

    
        supporting_legs = self.supporting_legs()
        foot_positions = []

        for leg_idx, is_supporting in enumerate(supporting_legs):
            if is_supporting:
                link_index = self.links[leg_idx, 2]  # Tibia link index
                pos, _ = self.client.getLinkState(self.hexId, link_index)[:2]
                foot_positions.append(pos[:2])  # Store only x, y coordinates

        if len(foot_positions) < 3:
            return False  # Not statically stable if fewer than 3 support points
        
        # Compute Center of Mass (CoM) projection
        com_pos, _ = self.client.getBasePositionAndOrientation(self.hexId)
        com_proj = [com_pos[0], com_pos[1], 0]  # Project CoM onto ground

        foot_positions = np.array(foot_positions)
        com = np.array(self.base_pos()[:2])  # Project CoM onto the ground (x, y only)

        # Remove previous lines
        for line_id in self.debug_lines:
            self.client.removeUserDebugItem(line_id)
        self.debug_lines.clear()

        # Draw CoM Projection
        com_line_id = self.client.addUserDebugLine(com_pos, com_proj, lineColorRGB=[1, 0, 0], lineWidth=2)  # Red vertical line
        self.debug_lines.append(com_line_id)  # Store the ID of the CoM line
        
        from scipy.spatial import ConvexHull
        # Compute convex hull and visualize it
        hull = ConvexHull(foot_positions)
        hull_path = foot_positions[hull.vertices]

        # Visualize the convex hull by drawing lines between the vertices
        for i in range(len(hull.vertices)):
            start_point = hull_path[i]
            end_point = hull_path[(i + 1) % len(hull.vertices)]  # Wrap around to connect last to first
            line_id = self.client.addUserDebugLine(start_point.tolist() + [0], end_point.tolist() + [0], lineColorRGB=[0, 1, 0], lineWidth=2)  # Green lines
            self.debug_lines.append(line_id)  # Store the ID of the hull line
        
        def point_in_hull(point, hull_points):
            from matplotlib.path import Path
            return Path(hull_points).contains_point(point)

        return point_in_hull(com, hull_path)

def supporting_legs(self):
    """Determines which legs of the hexapod are in contact with the ground or other objects."""
    
    tibia_links = self.links[:, 2]  # Extract tibia link indices
    contact_points = self.client.getContactPoints(self.hexId)  # Get all contact points for the hexapod

    contact_links = []
    for cp in contact_points:
        if cp[3] in tibia_links and cp[2] != self.hexId:  # Check if the link is a tibia and touching something external
            contact_links.append(cp[3])

    contact_links = np.array(contact_links)
    supporting_legs = np.isin(tibia_links, contact_links)  # Boolean mask for supporting legs

    return supporting_legs

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

def generate_turning_gait(base_radius=0.1, z_clearance=0.05, turn_rate=0.0):

# Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Get pressed keys
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        crab_angle -= np.radians(3)
        crab_angle = np.clip(crab_angle, -np.radians(60), np.radians(60))
        controller.update_crab_angle(crab_angle)

    elif keys[pygame.K_RIGHT]:
        crab_angle += np.radians(3)
        crab_angle = np.clip(crab_angle, -np.radians(60), np.radians(60))
        controller.update_crab_angle(crab_angle)