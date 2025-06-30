from enum import Enum
import numpy as np
# import sympy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

def rotate_point_2d(pt, pt_c, theta_deg):
    """
    Rotates a 2D point around another point using vector-matrix multiplication.

    Parameters:
        pt (tuple or np.ndarray): The point to rotate, shape (2,).
        theta_deg (float): Rotation angle in degrees.
        pt_c (tuple or np.ndarray): The center of rotation, shape (2,).

    Returns:
        np.ndarray: The rotated point, shape (2,).
    """
    # Convert to NumPy arrays
    pt = np.array(pt, dtype=float)
    pt_c = np.array(pt_c, dtype=float)

    # Convert angle to radians
    theta = np.radians(theta_deg)

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Shift point relative to center, rotate, then shift back
    rotated_pt = R @ (pt - pt_c) + pt_c

    return rotated_pt

# enum class defining robot motion states
class State(Enum):
    RW = 1
    CL = 2

class Motion(Enum):
    Forward = 1
    Rotation = 2

class UshapedRobot():
    """docstring for ClassName"""
    def __init__(self, a=1.0, b=1.0, c=1.0, d=1.0, m=1.0, s=1.0,fp = 1.0,initial_robot_position=[0.0,0.0],initial_robot_orientation = 0.0, dt=0.1, cavity_pumps_enabled=False,cavity_sensors_enabled=False):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.m = m
        self.s = s
        self.fp = fp
        self.orientation = initial_robot_orientation
        self.position = np.array(initial_robot_position)
        self.time = 0.0
        self.dt = dt
        self.r_cm = np.array(self.compute_CoM(a,b,c,d,s))
        
        self.n_modules = 2*a*(b+d) + c*d
        self.modules = self.module_positions(a, b, c, d)
        
        self.total_mass = self.m * self.n_modules
        
        
        self.I = self.compute_I(a,b,c,d,s,m)
        self.face_to_angle = {'T': np.pi / 2, 'B': -np.pi / 2, 'L': np.pi, 'R': 0.0}
        self.cavity_pumps_enabled = cavity_pumps_enabled
        self.cavity_sensors_enabled = cavity_sensors_enabled
        # self.External_Faces = self.find_external_faces()
        # print(self.External_Faces)
        self.pumps = self.define_external_pumps()
        self.sensors = self.define_external_sensors()
        self.n_pumps = len(self.pumps)
        self.force = np.array([0.0,0.0])
        self.velocity = np.array([0.0,0.0])
        self.acceleration = np.array([0.0,0.0])

        self.torque = None
        self.angular_velocity = 0.0
        self.angular_acceleration = 0.0

        self.state = State.RW
        self.motion = Motion.Forward
        self.forward_distance_to_travel = np.random.uniform(1.0, 50.0)  # Set initial forward distance
        self.travelled_distance = 0.0
        self.angle_to_rotate = 0.0
        self.rotated_angle = 0.0
        self.rotation_start_orientation = self.orientation  # Track orientation at start of rotation
        self.active_sensors = []

    def module_positions(self,a, b, c, d):
        modules = {}
        
        # left finger
        for i in range(1, a + 1):
            for j in range(d+1, d + b + 1):

                rect_bottom_left = self.position-self.r_cm + np.array([(i-1)*self.s,(j-1)*self.s])
                rect_top_left = rect_bottom_left + np.array([0,self.s])
                rect_bottom_right = rect_bottom_left + np.array([self.s,0])
                rect_top_right = rect_bottom_left + np.array([self.s,self.s])

                rect_bottom_left_rotated = rotate_point_2d(rect_bottom_left,self.position,self.orientation)
                rect_top_left_rotated = rotate_point_2d(rect_top_left,self.position,self.orientation)
                rect_bottom_right_rotated = rotate_point_2d(rect_bottom_right,self.position,self.orientation)
                rect_top_right_rotated = rotate_point_2d(rect_top_right,self.position,self.orientation)


                modules[(i, j)] = [rect_bottom_left_rotated,rect_bottom_right_rotated,rect_top_right_rotated,rect_top_left_rotated]

        # right finger
        for i in range(a + c + 1, 2 * a + c + 1):
            for j in range(d+1, d + b+1):
                rect_bottom_left = self.position-self.r_cm + np.array([(i-1)*self.s,(j-1)*self.s])
                rect_top_left = rect_bottom_left + np.array([0,self.s])
                rect_bottom_right = rect_bottom_left + np.array([self.s,0])
                rect_top_right = rect_bottom_left + np.array([self.s,self.s])

                rect_bottom_left_rotated = rotate_point_2d(rect_bottom_left,self.position,self.orientation)
                rect_top_left_rotated = rotate_point_2d(rect_top_left,self.position,self.orientation)
                rect_bottom_right_rotated = rotate_point_2d(rect_bottom_right,self.position,self.orientation)
                rect_top_right_rotated = rotate_point_2d(rect_top_right,self.position,self.orientation)

                modules[(i, j)] = [rect_bottom_left_rotated,rect_bottom_right_rotated,rect_top_right_rotated,rect_top_left_rotated]

        # Base
        for i in range(1, 2 * a + c +  1):
            for j in range(1, d+1):

                rect_bottom_left = self.position-self.r_cm + np.array([(i-1)*self.s,(j-1)*self.s])
                rect_top_left = rect_bottom_left + np.array([0,self.s])
                rect_bottom_right = rect_bottom_left + np.array([self.s,0])
                rect_top_right = rect_bottom_left + np.array([self.s,self.s])

                rect_bottom_left_rotated = rotate_point_2d(rect_bottom_left,self.position,self.orientation)
                rect_top_left_rotated = rotate_point_2d(rect_top_left,self.position,self.orientation)
                rect_bottom_right_rotated = rotate_point_2d(rect_bottom_right,self.position,self.orientation)
                rect_top_right_rotated = rotate_point_2d(rect_top_right,self.position,self.orientation)


                modules[(i, j)] = [rect_bottom_left_rotated,rect_bottom_right_rotated,rect_top_right_rotated,rect_top_left_rotated]

        return modules

    def compute_CoM(self,a,b,c,d,s):
        return [s * (a + c / 2.0),s * ((2*a+c)*(d**2) + 2*a*b*(2*d+b))/(2*((2*a+c)*d + 2*a*b))]

    def compute_I(self,a,b,c,d,s,m):
        return sum((1.0/6) * m * s**2 + m * np.sum((np.array([(i - 0.5) * s, (j - 0.5) * s]) - self.r_cm)**2) for (i, j) in self.modules)

    def pump_direction(self,S):
        return {'L': np.array([-1, 0]), 'R': np.array([1, 0]),
                'T': np.array([0, 1]), 'B': np.array([0, -1])}[S]

    def define_external_pumps(self):
        ext_all = []

        # no-cavity external 
        ext_all += [(1, j, 'L') for j in range(1,b + d + 1)]
        ext_all += [(i, 1, 'B') for i in range(1,2*a + c + 1)]
        ext_all += [(2*a + c, j, 'R') for j in range(1, b + d + 1)]
        ext_all += [(i, b + d, 'T') for i in range(1,a+1)]
        ext_all += [(i, b + d, 'T') for i in range(a + c + 1, 2*a + c + 1)]

        if self.cavity_pumps_enabled:
            ext_all += [(i, d, 'T') for i in range(a+1, a + c + 1)]
            ext_all += [(a, j, 'R') for j in range(d+1,b + d + 1)]
            ext_all += [(a+c+1, j, 'L') for j in range(d+1,b + d + 1)]

        return {location:0.0 for location in ext_all}

    def define_external_sensors(self):
        ext_all = []

        # no-cavity external 
        ext_all += [(1, j, 'L') for j in range(1,b + d + 1)]
        ext_all += [(i, 1, 'B') for i in range(1,2*a + c + 1)]
        ext_all += [(2*a + c, j, 'R') for j in range(1, b + d + 1)]
        ext_all += [(i, b + d, 'T') for i in range(1,a+1)]
        ext_all += [(i, b + d, 'T') for i in range(a + c + 1, 2*a + c + 1)]

        if self.cavity_sensors_enabled:
            ext_all += [(i, d, 'T') for i in range(a+1, a + c + 1)]
            ext_all += [(a, j, 'R') for j in range(d+1,b + d + 1)]
            ext_all += [(a+c+1, j, 'L') for j in range(d+1,b + d + 1)]

        return {location:0.0 for location in ext_all}

    def pump_position(self,i,j,S):
        return [(i - self.s/2) * self.s, (j - self.s/2) * self.s] + self.pump_direction(S) * (self.s / 2) 

    def activate_sensors(self,active_sensors):
        for sensor in active_sensors:
            self.sensors[sensor] = 1.0

    def activate_pumps(self,active_pumps):
        for pump in active_pumps:
            self.pumps[pump] = 1.0

    def compute_force(self):
        # Compute net force in robot local frame
        force_local = np.array([0.0, 0.0])
        for (i, j, S) in self.pumps:
            force_local += -self.pump_direction(S) * self.fp * self.pumps[(i, j, S)]
        # Rotate force to world frame
        theta = np.radians(self.orientation)
        c, s_ = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        self.force = R @ force_local

    def compute_torque(self):
        # print(f"[DIAG] compute_torque: orientation = {self.orientation} deg, {np.radians(self.orientation)} rad")
        self.torque = 0.0
        for (i, j, S), activation in self.pumps.items():
            if activation == 0.0:
                continue

            # Centre of module in local frame
            r_mod_centre = np.array([(i - 0.5) * self.s, (j - 0.5) * self.s])
            offset = self.pump_direction(S) * (self.s / 2)
            r_local = r_mod_centre + offset - self.r_cm

            # Rotate into world coordinates
            c, s_ = np.cos(np.radians(self.orientation)), np.sin(np.radians(self.orientation))
            R = np.array([[c, -s_], [s_, c]])
            r_world = R @ r_local
            force_world = R @ (self.fp * activation * -self.pump_direction(S))

            # 2D scalar torque = r_x * F_y - r_y * F_x
            self.torque += r_world[0] * force_world[1] - r_world[1] * force_world[0]

    def update_modules_position(self):
        # print(f"[DIAG] update_modules_position: orientation = {self.orientation} deg, {np.radians(self.orientation)} rad")
        for module in self.modules:
            i = module[0]
            j = module[1]

            rect_bottom_left = self.position-self.r_cm + np.array([(i-1)*self.s,(j-1)*self.s])
            rect_top_left = rect_bottom_left + np.array([0,self.s])
            rect_bottom_right = rect_bottom_left + np.array([self.s,0])
            rect_top_right = rect_bottom_left + np.array([self.s,self.s])

            rect_bottom_left_rotated = rotate_point_2d(rect_bottom_left,self.position,self.orientation)
            rect_top_left_rotated = rotate_point_2d(rect_top_left,self.position,self.orientation)
            rect_bottom_right_rotated = rotate_point_2d(rect_bottom_right,self.position,self.orientation)
            rect_top_right_rotated = rotate_point_2d(rect_top_right,self.position,self.orientation)
            self.modules[(i, j)] = [rect_bottom_left_rotated,rect_bottom_right_rotated,rect_top_right_rotated,rect_top_left_rotated]

    def increment_time(self):
        self.time += self.dt

    def sensor_angle(self, sensor):
        i, j, face = sensor
        # Center of the module in local frame
        module_center = np.array([(i - 0.5) * self.s, (j - 0.5) * self.s])
        face_offset = self.pump_direction(face) * (self.s / 2)
        sensor_local = module_center + face_offset
        # Rotate to world frame
        c, s_ = np.cos(np.radians(self.orientation)), np.sin(np.radians(self.orientation))
        R = np.array([[c, -s_], [s_, c]])
        sensor_world = R @ (sensor_local - self.r_cm) + self.position
        # Vector from CoM to sensor
        vec = sensor_world - self.position
        angle = np.arctan2(vec[1], vec[0])
        return angle

    def angle_from_heading(self, sensor, sensor_fov_angle=0):
        # Compute sensor position in local frame
        i, j, face = sensor
        module_center = np.array([(i - 0.5) * self.s, (j - 0.5) * self.s])
        face_offset = self.pump_direction(face) * (self.s / 2)
        sensor_local = module_center + face_offset
        # Heading vector in local frame: from CoM to (s*(a+c/2), b+d)
        heading_point = np.array([self.s * (self.a + self.c / 2.0), self.b + self.d])
        heading_vec = heading_point - self.r_cm
        # Vector from CoM to sensor in local frame
        sensor_vec = sensor_local - self.r_cm
        # Angle between heading_vec and sensor_vec
        angle = np.arctan2(sensor_vec[1], sensor_vec[0]) - np.arctan2(heading_vec[1], heading_vec[0])
        # Normalize to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        # Debug print for angle verification
        # Also print world-frame heading vector for diagnostic
        c, s_ = np.cos(np.radians(self.orientation)), np.sin(np.radians(self.orientation))
        R = np.array([[c, -s_], [s_, c]])
        heading_vec_world = R @ heading_vec
        # print(f"[DEBUG] angle_from_heading: sensor={sensor}, angle(rad)={angle:.4f}, angle(deg)={np.degrees(angle):.2f}, heading_vec={heading_vec}, sensor_vec={sensor_vec}, heading_vec_world={heading_vec_world}, orientation={self.orientation:.2f}")
        return angle
                
    def circular_mean(self, angles):
        """Compute the mean of angles (in radians), correctly handling wraparound."""
        return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    def compute_angle_to_rotate(self, active_sensors, sensor_fov_angle=0):
        """
        Divide active_sensors into left and right sets based on their i index.
        For each sensor, compute the angle from the robot's heading (not just CoM) to the sensor's position,
        including a field-of-view offset.
        Return the average of the set (left or right) with the bigger cardinality.
        If equal, return the average of the left set.
        Log the computed angles for debugging.
        """
        left_sensors = [sensor for sensor in active_sensors if sensor[0] <= self.a + self.c]
        right_sensors = [sensor for sensor in active_sensors if sensor[0] > self.a + self.c]

        if len(left_sensors) > len(right_sensors):
            angles = [self.angle_from_heading(sensor, sensor_fov_angle) for sensor in left_sensors]
            return self.circular_mean(angles)
        elif len(right_sensors) > len(left_sensors):
            angles = [self.angle_from_heading(sensor, sensor_fov_angle) for sensor in right_sensors]
            return self.circular_mean(angles)
        else:
            angles = [self.angle_from_heading(sensor, sensor_fov_angle) for sensor in left_sensors]
            return self.circular_mean(angles)

    def controller(self):
        # Update state and motion based on sensors and current state:        
        if self.state == State.RW:
            if len(self.active_sensors) > 0:
                # print(f"[STATE] Switching to State.CL at time {self.time:.2f}, detected sensors: {self.active_sensors}")
                self.state = State.CL
                self.motion = Motion.Rotation
                self.angle_to_rotate = np.degrees(self.compute_angle_to_rotate(self.active_sensors, sensor_fov_angle))
                # print(f"[ROTATE] Set angle_to_rotate (sensor-based): {self.angle_to_rotate:.2f} deg at time {self.time:.2f}")
                self.rotated_angle = 0.0
                self.rotation_start_orientation = self.orientation
            else:
                if self.motion == Motion.Forward:
                    if self.travelled_distance >= self.forward_distance_to_travel:
                        self.motion = Motion.Rotation
                        self.angle_to_rotate = np.degrees(np.random.uniform(-np.pi, np.pi)) # Random rotation angle in degrees
                        # print(f"[ROTATE] Set random angle_to_rotate: {self.angle_to_rotate:.2f} deg at time {self.time:.2f}")
                        self.rotated_angle = 0.0
                        self.rotation_start_orientation = self.orientation
                else: #self.motion == Motion.Rotation
                    # Compute signed rotated angle
                    delta = self.orientation - self.rotation_start_orientation
                    # Normalize to [-180, 180]
                    delta = (delta + 180) % 360 - 180
                    self.rotated_angle = delta
                    # print(f"[ROTATE] Rotating: rotated_angle={self.rotated_angle:.2f} deg, angle_to_rotate={self.angle_to_rotate:.2f} deg at time {self.time:.2f}")
                    if (self.angle_to_rotate >= 0 and self.rotated_angle >= self.angle_to_rotate) or (self.angle_to_rotate < 0 and self.rotated_angle <= self.angle_to_rotate):
                        # print(f"[DEBUG] Rotation complete (RW): intended={self.angle_to_rotate:.2f} deg, actual={self.rotated_angle:.2f} deg, start={self.rotation_start_orientation:.2f} deg, end={self.orientation:.2f} deg")
                        self.angular_velocity = 0.0  # Stop further rotation after completion
                        # print(f"[ROTATE] Rotation complete at time {self.time:.2f}")
                        self.motion = Motion.Forward
                        self.travelled_distance = 0.0
                        self.angle_to_rotate = 0.0
                # Ensure consistent state
                if self.motion == Motion.Forward:
                    self.state = State.RW
        else: # self.state == State.CL
            if self.motion == Motion.Rotation:
                # Compute signed rotated angle
                delta = self.orientation - self.rotation_start_orientation
                delta = (delta + 180) % 360 - 180
                self.rotated_angle = delta
                if (self.angle_to_rotate >= 0 and self.rotated_angle >= self.angle_to_rotate) or (self.angle_to_rotate < 0 and self.rotated_angle <= self.angle_to_rotate):
                    # print(f"[DEBUG] Rotation complete (CL): intended={self.angle_to_rotate:.2f} deg, actual={self.rotated_angle:.2f} deg, start={self.rotation_start_orientation:.2f} deg, end={self.orientation:.2f} deg")
                    self.angular_velocity = 0.0  # Stop further rotation after completion
                    self.angular_acceleration = 0.0  # Also zero angular acceleration
                    self.motion = Motion.Forward
                    self.travelled_distance = 0.0
                    self.forward_distance_to_travel = 10.0
            else: # self.motion == Motion.Forward
                if self.travelled_distance >= self.forward_distance_to_travel:
                    self.state = State.RW
                    self.motion = Motion.Forward
                    self.travelled_distance = 0.0
        # Reset all pumps to 0.0 before activating the required ones
        for key in self.pumps:
            self.pumps[key] = 0.0

        if self.motion == Motion.Forward:
            # Forward motion: activate pumps to move forward
            for i in range(1, 2*self.a + self.c + 1):
                self.pumps[(i, 1, 'B')] = 1.0
        else: # self.motion == Motion.Rotation
            # Choose rotation direction based on sign of angle_to_rotate
            if self.angle_to_rotate >= 0:
                # print(f"[PUMPS] Rotating anticlockwise (angle_to_rotate={self.angle_to_rotate:.2f}): activating B, L, R, T pumps")
                self.pumps[(2*self.a + self.c,1,'B')] = 1.0
                self.pumps[(1,1,'L')] = 1.0
                self.pumps[(2*self.a + self.c, self.b+self.d,'R')] = 1.0
                self.pumps[(1,self.b + self.d,'T')] = 1.0
            else:
                # print(f"[PUMPS] Rotating clockwise (angle_to_rotate={self.angle_to_rotate:.2f}): activating R, B, T, L pumps")
                self.pumps[(2*self.a + self.c,1,'R')] = 1.0
                self.pumps[(1,1,'B')] = 1.0
                self.pumps[(2*self.a + self.c, self.b+self.d,'T')] = 1.0
                self.pumps[(1,self.b + self.d,'L')] = 1.0

        # Control for moving forward:
        # for i in range(1,2*self.a + self.c + 1):
        #     self.robot.pumps[(i,1,'B')] = 1.0

        # Control for rotating on spot
        # self.pumps[(2*self.a + self.c,1,'B')] = 1.0
        # self.pumps[(1,1,'L')] = 1.0
        # self.pumps[(2*self.a + self.c, self.b+self.d,'R')] = 1.0
        # self.pumps[(1,self.b + self.d,'T')] = 1.0

    def update_position(self):
        prev_position = self.position.copy()
        prev_orientation = self.orientation
        self.position += self.velocity * self.dt + 0.5*self.acceleration * (self.dt**2)
        self.orientation += np.degrees(self.angular_velocity * self.dt)  # orientation always in degrees
        # Normalize orientation to [0, 360)
        self.orientation = self.orientation % 360
        self.update_modules_position()
        self.log_orientation_for_debug()  # Log after update, before plotting

        # Update travelled distance
        self.travelled_distance += np.linalg.norm(self.position - prev_position)
        # self.rotated_angle is now handled in controller using signed difference

        self.velocity = self.acceleration * self.dt
        self.angular_velocity += self.angular_acceleration * self.dt

        # Apply angular damping to simulate friction
        damping = 0.5  # adjust as needed
        self.angular_velocity *= (1 - damping)

        # compute force
        self.compute_force()
        self.acceleration = self.force/self.total_mass
        
        #compute torque        
        self.compute_torque()
        # Zero out tiny torques to prevent numerical drift before computing angular acceleration
        if abs(self.torque) < 1e-8:
            self.torque = 0.0
        self.angular_acceleration = self.torque / self.I

        # Zero out tiny angular velocities to prevent numerical drift
        if abs(self.angular_velocity) < 1e-8:
            self.angular_velocity = 0.0

    def log_orientation_for_debug(self):
        # Improved debug log for rotation and orientation
        msg = f"[VISUAL-DEBUG] time={self.time:.2f}s | orientation={self.orientation:.2f} deg | angular_velocity={self.angular_velocity:.4f} deg/s"
        if hasattr(self, 'rotation_start_orientation') and hasattr(self, 'angle_to_rotate') and hasattr(self, 'rotated_angle'):
            intended_end = (self.rotation_start_orientation + self.angle_to_rotate) % 360
            msg += (f" | rotation_start={self.rotation_start_orientation:.2f} deg"
                    f" | intended_delta={self.angle_to_rotate:.2f} deg"
                    f" | actual_delta={self.rotated_angle:.2f} deg"
                    f" | expected_end={intended_end:.2f} deg")
        msg += f" | motion={self.motion.name} | state={self.state.name}"
        # print(msg)

    def check_sensors_readings(self, objects, sensor_fov_angle, sensor_range):
        """
        Efficiently check if any part of an object is within the sensor's FoV and range.
        Uses a fast geometric check: for each object, project the center onto the sensor axis and check perpendicular distance.
        For nonzero FoV, also check the angle span as before.
        """
        self.active_sensors = []
        robot_pos = self.position
        for sensor in self.sensors:
            i, j, face = sensor
            # Sensor position in local frame: module center + offset toward face
            module_center = np.array([(i - 0.5) * self.s, (j - 0.5) * self.s])
            face_offset = self.pump_direction(face) * (self.s / 2)
            sensor_local = module_center + face_offset
            c, s_ = np.cos(np.radians(self.orientation)), np.sin(np.radians(self.orientation))
            R = np.array([[c, -s_], [s_, c]])
            sensor_pos = R @ (sensor_local - self.r_cm) + robot_pos
            # Sensor direction in world frame (unit vector)
            face_angle = self.face_to_angle[face]
            sensor_angle = np.radians(self.orientation) + face_angle
            sensor_dir = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
            # Print for debugging direction
            # print(f"[SENSOR-DIR-DEBUG] sensor={sensor} face={face} face_angle={np.degrees(face_angle):.2f} orientation={self.orientation:.2f} sensor_angle={np.degrees(sensor_angle):.2f} sensor_dir={sensor_dir}")
            found = False
            for obj in objects:
                obj_center = np.array(obj['center'])
                obj_radius = obj['radius']
                # Vector from sensor to object center
                vec = obj_center - sensor_pos
                dist = np.linalg.norm(vec)
                # DEBUG PRINTS
                # print(f"[SENSOR-DEBUG] sensor={sensor} pos={sensor_pos} dir={sensor_dir} obj_center={obj_center} dist={dist:.3f}")
                if dist > sensor_range + obj_radius:
                    continue
                # Project vec onto sensor direction
                proj = np.dot(vec, sensor_dir)
                # Perpendicular distance from object center to sensor axis
                perp = np.linalg.norm(vec - proj * sensor_dir)
                # For very small FoV, treat as a ray with width = object radius
                if sensor_fov_angle <= 1e-3:
                    if 0 < proj < sensor_range and perp <= obj_radius:
                        # print(f"[SENSOR-DEBUG] --> DETECTED (ray mode): proj={proj:.3f}, perp={perp:.3f}")
                        found = True
                        break
                else:
                    # For nonzero FoV, use the angle span method
                    obj_angle = np.arctan2(vec[1], vec[0])
                    if dist > obj_radius:
                        angle_span = np.arcsin(obj_radius / dist)
                    else:
                        angle_span = np.pi
                    fov_half = np.radians(sensor_fov_angle / 2)
                    angle_diff = np.arctan2(np.sin(obj_angle - sensor_angle), np.cos(obj_angle - sensor_angle))
                    # print(f"[SENSOR-DEBUG]    obj_angle={np.degrees(obj_angle):.2f} sensor_angle={np.degrees(sensor_angle):.2f} angle_diff={np.degrees(angle_diff):.2f} angle_span={np.degrees(angle_span):.2f} fov_half={np.degrees(fov_half):.2f}")
                    # Robust overlap check:
                    if abs(angle_diff) <= fov_half + angle_span:
                        # print(f"[SENSOR-DEBUG] --> DETECTED (fov mode, robust): angle_diff={np.degrees(angle_diff):.2f}, angle_span={np.degrees(angle_span):.2f}")
                        found = True
                        break
            self.sensors[sensor] = 1.0 if found else 0.0
            if found:
                self.active_sensors.append(sensor)

    def is_object_in_cavity(self, obj_center, obj_radius):
        """
        Returns True if the entire object (center and radius) is inside the robot's cavity.
        The cavity is the rectangle between the two fingers, above the base.
        """
        # Cavity bounds in robot local frame
        x_min = self.s * (self.a)
        x_max = self.s * (self.a + self.c)
        y_min = self.s * (self.d)
        y_max = self.s * (self.d + self.b)
        # Correct transformation: rotate (obj - robot_pos), then add r_cm
        c, s_ = np.cos(np.radians(-self.orientation)), np.sin(np.radians(-self.orientation))
        R = np.array([[c, -s_], [s_, c]])
        local_center = R @ (np.array(obj_center) - self.position) + self.r_cm
        epsilon = 1e-6
        # Print robot and object info for every check
        # print(f"[CAVITY-DEBUG] Robot pos={self.position}, orient={self.orientation:.2f} deg")
        # print(f"[CAVITY-DEBUG] Object world={obj_center}, local={local_center}, radius={obj_radius}")
        # print(f"[CAVITY-DEBUG] Cavity x:[{x_min},{x_max}], y:[{y_min},{y_max}] (s={self.s}, a={self.a}, b={self.b}, c={self.c}, d={self.d})")
        # print(f"[CAVITY-DEBUG] x_dist_left={local_center[0] - (x_min + obj_radius):.6f}, x_dist_right={(x_max - obj_radius) - local_center[0]:.6f}, y_dist_bottom={local_center[1] - (y_min + obj_radius):.6f}, y_dist_top={(y_max - obj_radius) - local_center[1]:.6f}")
        # Check if the entire object is inside the cavity rectangle (with margin for radius and epsilon)
        if (x_min + obj_radius - epsilon <= local_center[0] <= x_max - obj_radius + epsilon and
            y_min + obj_radius - epsilon <= local_center[1] <= y_max - obj_radius + epsilon):
            # print(f"[CAVITY-DEBUG] Object at {obj_center} (local {local_center}) is INSIDE cavity.")
            return True
        # print(f"[CAVITY-DEBUG] Object at {obj_center} (local {local_center}) is OUTSIDE cavity.")
        return False

class Simulation(object):
    """docstring for ClassName"""
    def __init__(self, a=1, b=1, c=1, d=1, s=1.0, m = 1.0,fp=1.0, env_size=[100.0,100.0], initial_robot_position=[0.0,0.0], initial_robot_orientation = 0.0, dt=0.1, n_objects=5, object_radius=2.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.s = s
        self.m = m
        self.fp = fp
        self.robot = UshapedRobot(a, b, c, d,s,m,fp,initial_robot_position,initial_robot_orientation,dt)
        self.env_size = env_size
        self.time = 0.0
        self.fig = plt.figure(figsize=tuple(self.env_size))
        self.ax = plt.gca()
        self.dt = dt
        self.n_objects = n_objects
        self.object_radius = object_radius
        self.objects = self.spawn_objects(n_objects, object_radius)
        self.collected_objects = 0

    def spawn_objects(self, n, radius):
        # Always use random spawning for all n
        objects = []
        margin = radius + 1.0
        for _ in range(n):
            x = np.random.uniform(margin, self.env_size[0] - margin)
            y = np.random.uniform(margin, self.env_size[1] - margin)
            objects.append({'center': (x, y), 'radius': radius})
        return objects

    def step(self):
        # Efficiently update sensor readings based on objects in the environment
        self.robot.check_sensors_readings(self.objects, sensor_fov_angle, sensor_range)
        self.robot.controller() # set pump activations
        self.robot.update_position() # move the robot according to it's current pump actiations
        self.robot.increment_time()
        # Object collection logic
        remaining_objects = []
        for obj in self.objects:
            if self.robot.is_object_in_cavity(obj['center'], obj['radius']):
                self.collected_objects += 1
                # print(f"[COLLECT] Object collected at {obj['center']}, total: {self.collected_objects}")
            else:
                remaining_objects.append(obj)
        self.objects = remaining_objects
        # print(f"State: {self.robot.state}, Motion: {self.robot.motion}, Travelled: {self.robot.travelled_distance:.2f}, Rotated: {self.robot.rotated_angle:.2f}, Torque: {self.robot.torque}, Force: {self.robot.force}, Velocity: {self.robot.velocity}, Acceleration: {self.robot.acceleration}, Angular Velocity: {self.robot.angular_velocity}")

    def visualise(self):
        self.ax.clear()  # clear previous patches and points

        # Draw modules
        for module in self.robot.modules.values():
            polygon = Polygon(module, closed=True, facecolor='white', edgecolor='black')
            self.ax.add_patch(polygon)

        # Draw CoM (black)
        self.ax.scatter(self.robot.position[0], self.robot.position[1], color='black', s=6)

        # Draw circular objects (blue)
        for obj in self.objects:
            circ = Circle(obj['center'], obj['radius'], color='blue', alpha=0.5)
            self.ax.add_patch(circ)

        # Draw FoV sensors
        for sensor in self.robot.sensors:
            i, j, face = sensor
            # Sensor position in local frame (module center + face offset)
            module_center = np.array([(i - 0.5) * self.robot.s, (j - 0.5) * self.robot.s])
            face_offset = self.robot.pump_direction(face) * (self.robot.s / 2)
            sensor_local = module_center + face_offset
            c, s_ = np.cos(np.radians(self.robot.orientation)), np.sin(np.radians(self.robot.orientation))
            R = np.array([[c, -s_], [s_, c]])
            sensor_pos_world = R @ (sensor_local - self.robot.r_cm) + self.robot.position
            # Sensor direction in world frame (unit vector)
            face_angle = self.robot.face_to_angle[face]
            sensor_angle = np.radians(self.robot.orientation) + face_angle
            # Draw FoV as a wedge
            theta1 = np.degrees(sensor_angle - np.radians(sensor_fov_angle/2))
            theta2 = np.degrees(sensor_angle + np.radians(sensor_fov_angle/2))
            is_active = sensor in self.robot.active_sensors
            wedge_color = 'red' if is_active else 'green'
            wedge_alpha = 0.3 if is_active else 0.2
            wedge = plt.matplotlib.patches.Wedge(
                sensor_pos_world, sensor_range, theta1, theta2, color=wedge_color, alpha=wedge_alpha
            )
            self.ax.add_patch(wedge)

        # Draw cavity as a rectangle (yellow, semi-transparent)
        # Cavity bounds in robot local frame
        x_min = self.robot.s * (self.robot.a)
        x_max = self.robot.s * (self.robot.a + self.robot.c)
        y_min = self.robot.s * (self.robot.d)
        y_max = self.robot.s * (self.robot.d + self.robot.b)
        # Rectangle corners in local frame
        cavity_corners_local = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        # Transform to world frame
        c, s_ = np.cos(np.radians(self.robot.orientation)), np.sin(np.radians(self.robot.orientation))
        R = np.array([[c, -s_], [s_, c]])
        cavity_corners_world = [R @ (np.array(corner) - self.robot.r_cm) + self.robot.position for corner in cavity_corners_local]
        cavity_polygon = Polygon(cavity_corners_world, closed=True, facecolor='yellow', edgecolor='orange', alpha=0.2, linewidth=2)
        self.ax.add_patch(cavity_polygon)

        # Draw heading vector (red arrow from CoM)
        heading_point = np.array([self.robot.s * (self.robot.a + self.robot.c / 2.0), self.robot.b + self.robot.d])
        heading_vec = heading_point - self.robot.r_cm
        c, s_ = np.cos(np.radians(self.robot.orientation)), np.sin(np.radians(self.robot.orientation))
        R = np.array([[c, -s_], [s_, c]])
        heading_vec_world = R @ heading_vec
        start = self.robot.position
        end = self.robot.position + heading_vec_world
        self.ax.arrow(start[0], start[1], heading_vec_world[0], heading_vec_world[1], head_width=0.5, head_length=1.0, fc='red', ec='red', linewidth=2, length_includes_head=True, zorder=10)

        # Draw expected heading vector (blue arrow from CoM) if in rotation mode
        if hasattr(self.robot, 'rotation_start_orientation') and hasattr(self.robot, 'angle_to_rotate'):
            expected_end_orientation = (self.robot.rotation_start_orientation + self.robot.angle_to_rotate) % 360
            c_exp, s_exp = np.cos(np.radians(expected_end_orientation)), np.sin(np.radians(expected_end_orientation))
            R_exp = np.array([[c_exp, -s_exp], [s_exp, c_exp]])
            heading_vec_expected = R_exp @ heading_vec
            self.ax.arrow(start[0], start[1], heading_vec_expected[0], heading_vec_expected[1], head_width=0.5, head_length=1.0, fc='blue', ec='blue', linewidth=2, length_includes_head=True, zorder=10)

        # Set limits and aspect ratio
        self.ax.set_xlim(0, self.env_size[0])
        self.ax.set_ylim(0, self.env_size[1])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title(f"Time: {self.time:.2f}s")

        plt.pause(0.01)  # brief pause to update the figure
        

    
# Set a fixed random seed for reproducibility
np.random.seed(42)
# Robot parameters
a, b, c, d = 2, 5, 3, 2
s,m,fp = 1.0, 1.0, 1.0
env_size=[100.0,100.0]
initial_robot_position=[50.0,50.0]
initial_robot_orientation = 0.0
experiment_time = 1000.0
dt = 1.0
n_objects = 50  # Restore to 10 random objects
object_radius = 0.5  # Set the radius of the objects
sensor_fov_angle = 30  # degrees
sensor_range = 3.0    # units
simulation = Simulation(a, b, c, d, s, m,fp, env_size, initial_robot_position, initial_robot_orientation,dt, n_objects=n_objects, object_radius=object_radius)
plt.ion()

while simulation.time <= experiment_time:
    # Exit loop if figure is closed
    if not plt.fignum_exists(simulation.fig.number):
        break

    simulation.step()
    simulation.visualise()
    simulation.time += dt

plt.ioff()  # turn off interactive mode when done
plt.show()  # final show to keep the window open at the end