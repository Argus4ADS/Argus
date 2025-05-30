import math

import numpy as np
import carla
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from agents.navigation.local_planner import RoadOption
from dstar_lite import DStarLite, Node


class RoutePlannerMR(object):
    """
      This class is used the experts navigation and provides not only the route but preprocesses it and provides useful
      information like the next stop sign and the next traffic light.
      """

    def __init__(self, config):
        """
            Initialize the RoutePlanner object.

            Args:
                config (GlobalConfig): Object of the config for hyperparameters.
            """

        self.config = config

        self.points_per_meter = self.config.points_per_meter
        self.ego_vehicles_route_point_search_distance = self.config.ego_vehicles_route_point_search_distance
        self.lane_shift_extension_length_for_yield_to_emergency_vehicle = \
            self.config.lane_shift_extension_length_for_yield_to_emergency_vehicle
        self.transition_smoothness_distance = self.config.transition_smoothness_distance
        self.route_shift_start_distance_invading_turn = self.config.route_shift_start_distance_invading_turn
        self.route_shift_end_distance_invading_turn = self.config.route_shift_end_distance_invading_turn
        self.fence_avoidance_margin_invading_turn = self.config.fence_avoidance_margin_invading_turn
        self.minimum_lane_width_threshold = self.config.minimum_lane_width_threshold
        self.speed_limit_waypoints_spacing_check = self.config.speed_limit_waypoints_spacing_check
        self.leading_vehicles_max_route_distance = self.config.leading_vehicles_max_route_distance
        self.leading_vehicles_max_route_angle_distance = self.config.leading_vehicles_max_route_angle_distance
        self.leading_vehicles_maximum_detection_radius = self.config.leading_vehicles_maximum_detection_radius
        self.trailing_vehicles_max_route_distance = self.config.trailing_vehicles_max_route_distance
        self.trailing_vehicles_max_route_distance_lane_change = \
            self.config.trailing_vehicles_max_route_distance_lane_change
        self.tailing_vehicles_maximum_detection_radius = self.config.tailing_vehicles_maximum_detection_radius
        self.max_distance_lane_change_trailing_vehicles = self.config.max_distance_lane_change_trailing_vehicles
        self.extra_route_length = self.config.extra_route_length

        self.route_waypoints = []
        self.route_points = np.array([[]])
        self.original_route_points = np.array([[]])
        self.commands = []
        self.rotation_angles = []

        self.distances_to_next_stop_signs = np.array([])
        self.next_stop_signs = []

        self.distances_to_next_traffic_lights = np.array([])
        self.next_traffic_lights = []

        self.route_index = 0
        self.last_route_index = 0

        # ======= DstarLite with Astar
        self.dstar_lite = None  
        self.grid_resolution = 1
        self.local_map_max_distance = 40.0
        self.grid_width = 80
        self.grid_height = 80

    def save(self):
        """
            Save the current route index location, which could be saved before forecasting the ego vehicle.
            """
        self.last_route_index = self.route_index

    def load(self):
        """
            Load the previously saved route index location, which could be used for forecasting the ego vehicle.
            """
        self.route_index = self.last_route_index

    def run_step(self, agent_position):
        """
            Update the route index based on the agent's current position and retrieve relevant information at that index.

            Args:
                agent_position (numpy.ndarray): Current location of the agent.
            """
        till = self.ego_vehicles_route_point_search_distance
        search_range = min(self.route_index + till, self.route_points.shape[0])

        # Find the index of the nearest route point to the agent's position
        self.route_index += np.argmin(np.linalg.norm(agent_position[None, :2] -
                                                     self.route_points[self.route_index:search_range, :2], axis=1))

        return (self.route_points[self.route_index:], self.route_waypoints[self.route_index:],
                self.commands[self.route_index:], self.distances_to_next_traffic_lights[self.route_index],
                self.next_traffic_lights[self.route_index], self.distances_to_next_stop_signs[self.route_index],
                self.next_stop_signs[self.route_index])

    def get_closest_route_index(self, begin_idx, location):
        """
            Finds the index of the closest route point to a given location using gradient descent with constant gradient.

            Args:
                begin_idx (int): Starting index for the search.
                location (carla.Location): Location for which the closest route point is to be found.

            Returns:
                int: Index of the closest route point.
            """
        index = begin_idx
        location_np = np.array([location.x, location.y])

        # calculate the search direction
        direction = 1
        if np.linalg.norm(location_np - self.original_route_points[index, :2]) < np.linalg.norm(
                location_np - self.original_route_points[index + 1, :2]):
            direction = -1

        # The following is like a gradient descent with a constant gradient.
        while True:
            # check if we have reached the first or last route point
            if index + direction == 0 or index + direction == self.original_route_points.shape[0]:
                return index

            dist1 = np.linalg.norm(
                location_np - self.original_route_points[index, :2])
            dist2 = np.linalg.norm(
                location_np - self.original_route_points[index + direction, :2])
            # check if we have found the closest route point
            if dist1 < dist2:
                return index

            index += direction

    def setup_route(self, global_plan, carla_world, carla_map, starts_with_parking_exit, vehicle_loc):
        """
            Set up the route for the autonomous vehicle based on the given global plan.

            Args:
                global_plan (list): A list of (carla.Transform, carla.RoadOption) tuples representing the global plan.
                carla_world (carla.World): The CARLA world object.
                carla_map (carla.Map): The CARLA map object.
                starts_with_parking_exit (bool): A flag indicating if the route starts with a parking exit scenario.
                vehicle_location (carla.Location): The initial location of the vehicle.
            """
        self.route_index = self.extra_route_length * self.points_per_meter
        self.last_route_index = self.route_index
        self.carla_map = carla_map
        self.carla_world = carla_world

        # Get all waypoint objects of the route and add extra waypoints at the end
        # to ensure the vehicle completes the route properly and avoids unexpected side effects
        route_waypoints = [transform.location for transform, _ in global_plan]
        route_waypoints = [carla_map.get_waypoint(
            loc) for loc in route_waypoints]
        cmds = [cmd for _, cmd in global_plan]

        # Handle the case where the route starts with a parking exit scenario
        # In this case the first wp is on the center of the road, not the parking lot,
        # where the agent starts
        if starts_with_parking_exit:  # workaraound for ParkingExit scenario
            self.route_index = 0
            self.last_route_index = 0

            cmds.insert(0, RoadOption.CHANGELANELEFT)
            route_waypoints.insert(0, carla_map.get_waypoint(vehicle_loc))
        else:
            # Add extra waypoints at the beginning of the route
            for _ in range(self.extra_route_length):
                prev_wps = route_waypoints[0].previous(1)
                if len(prev_wps) == 0:
                    break
                route_waypoints.insert(0, prev_wps[0])
                cmds.insert(0, RoadOption.LANEFOLLOW)
                self.route_index += 1
                self.last_route_index += 1

        # Add extra waypoints at the end of the route
        for _ in range(self.extra_route_length):
            next_wps = route_waypoints[-1].next(1)
            if len(next_wps) == 0:
                break

            route_waypoints.append(next_wps[0])
            cmds.append(RoadOption.LANEFOLLOW)

        # Generate a numpy array containing the route locations
        route_points = [wp.transform.location for wp in route_waypoints]
        route_points = np.array([[loc.x, loc.y, loc.z]
                                 for loc in route_points])

        # Smooth and interpolate the route
        self.route_points, self.commands = self.smooth_and_supersample(
            route_points, cmds)
        self.original_route_points = np.copy(self.route_points)
        self.commands_orig = self.commands.copy()

        # Get the waypoint objects for the route points
        self.route_waypoints = []
        for route_loc in self.route_points:
            wp = carla_map.get_waypoint(carla.Location(
                x=route_loc[0], y=route_loc[1], z=route_loc[2]))
            self.route_waypoints.append(wp)

        self.compute_route_info(carla_world, carla_map)

    def compute_rotation_angles(self, route_points):
        """
            Computes the yaw angles corresponding to the ego vehicle's orientation at individual route points in degrees.

            Args:
                route_points (numpy.ndarray): Array containing the route points.

            Returns:
                numpy.ndarray: Array containing the yaw angles at each route point.
            """

        # Compute differences between consecutive route points
        indices = np.arange(1, route_points.shape[0] - 1)
        differences = route_points[indices + 1] - route_points[indices - 1]

        # Compute yaw angles in degrees
        yaws = np.arctan2(differences[:, 1], differences[:, 0]) * 180. / np.pi

        # Add first and last yaw angles to maintain array length
        yaws = np.concatenate([[yaws[0]], yaws, [yaws[-1]]])

        return yaws

    def smooth_and_supersample(self, original_route_points, commands):
        """
            Smooths and supersamples the given route to increase density and matches commands accordingly.

            Args:
                original_route_points (numpy.ndarray): Array containing the original route points.
                commands (list): List of commands corresponding to the route points.

            Returns:
                tuple: A tuple containing the smoothed and supersampled route points, and the updated commands.
            """

        # sample x points per number of route points for later
        num_supersample_per_point = 10
        # number of points to interpolate between each pair of original points
        num_samples = self.points_per_meter * num_supersample_per_point
        # Length of segments along the smoothed route
        segment_length = 1. / self.points_per_meter
        num_original_points = original_route_points.shape[0]

        # Create interpolation functions for each dimension
        interp_x = interp1d(np.arange(num_original_points),
                            original_route_points[:, 0])
        interp_y = interp1d(np.arange(num_original_points),
                            original_route_points[:, 1])
        interp_z = interp1d(np.arange(num_original_points),
                            original_route_points[:, 2])

        # Interpolate points along the original route
        x_supersampled = interp_x(
            np.arange(0, num_original_points - 1, 1. / num_samples))
        y_supersampled = interp_y(
            np.arange(0, num_original_points - 1, 1. / num_samples))
        z_supersampled = interp_z(
            np.arange(0, num_original_points - 1, 1. / num_samples))

        route_supersampled = np.column_stack(
            [x_supersampled, y_supersampled, z_supersampled])

        # Calculate cumulative distances along the supersampled route
        cumulative_distances = np.cumsum(np.linalg.norm(
            np.diff(route_supersampled, axis=0), axis=1))
        cumulative_distances = np.insert(cumulative_distances, 0, 0)
        cumulative_distances = cumulative_distances % segment_length

        # Find indices of points at segment boundaries
        segment_indices = np.insert(np.argwhere(
            cumulative_distances[1:] < cumulative_distances[:-1]), 0, 0)
        smoothed_points = route_supersampled[segment_indices]

        # Interpolate commands for the smoothed points
        num_original_commands = len(commands)
        command_indices = np.minimum(
            np.round(segment_indices.astype("float") /
                     self.points_per_meter / num_supersample_per_point),
            num_original_commands - 1).astype("int")
        smoothed_commands = np.array([commands[idx]
                                      for idx in command_indices])

        return smoothed_points, smoothed_commands

    def compute_route_info(self, carla_world, carla_map):
        """
            Computes additional information for the route such as distances to traffic lights and stop signs,
            speed limits, and prevents too early lane changes and computes yaw angles corresponding to the ego
            vehicle's orientation at individual route points in degrees.

            Args:
                carla_world: Carla world instance.
                carla_map: Carla map instance.
            """
        self.rotation_angles = self.compute_rotation_angles(self.route_points)
        self.compute_distances_to_traffic_lights(carla_world)
        self.compute_distances_to_stop_signs(carla_world, carla_map)
        self.prevent_too_early_lane_changes()

    def prevent_too_early_lane_changes(self):
        """
            Prevents too early lane changes by ensuring that the agent continues on the previous lane for a bit longer
            in case the lane is too narrow.
            """
        lane_threshold = self.minimum_lane_width_threshold

        # Iterate over route waypoints
        for i in range(len(self.route_waypoints) - 2):
            # Check that we have not reached the last waypoint and the lane width increases
            if self.route_waypoints[i + 1].lane_width < lane_threshold and self.route_waypoints[
                i + 2].lane_width < lane_threshold and self.route_waypoints[i + 1].lane_width < self.route_waypoints[
                i + 2].lane_width:
                j = i + 1
                to_left = self.commands[i] == RoadOption.CHANGELANELEFT

                # Continue on the previous lane until it's wide enough
                while True:
                    if j == len(self.route_waypoints) or self.route_waypoints[j].lane_width >= lane_threshold:
                        break

                    # Get the waypoint of the previous lane
                    wp = self.route_waypoints[j].get_right_lane() if to_left \
                        else self.route_waypoints[j].get_left_lane()
                    wp = self.route_waypoints[j] if wp is None else wp

                    # Update route waypoints and points
                    self.route_waypoints[j] = wp
                    self.route_points[j] = np.array(
                        [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z])
                    self.original_route_points[j] = np.array(
                        [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z])
                    j += 1

    def compute_distances_to_traffic_lights(self, carla_world):
        """
            Compute the distance to the next traffic light from each individual route location.

            Args:
                carla_world: Carla world instance.
            """
        # Initialize arrays to store distances and next traffic lights
        self.distances_to_next_traffic_lights = np.full(
            self.route_points.shape[0], np.inf)
        self.next_traffic_lights = [None] * self.route_points.shape[0]

        # Initialize variables
        next_traffic_light = None
        traffic_light_already_recorded = False
        distance_idx = np.inf

        # Iterate over route points in reverse order
        for i in range(len(self.route_points) - 1, -1, -1):
            waypoint = self.route_waypoints[i]
            traffic_lights = carla_world.get_traffic_lights_from_waypoint(
                waypoint, 5)

            # Check if the found traffic light was already recorded in the past
            if traffic_lights:
                if not traffic_light_already_recorded:
                    distance_idx = 0
                    next_traffic_light = traffic_lights[0]
                else:
                    distance_idx += 1

                traffic_light_already_recorded = True
            else:
                distance_idx += 1
                traffic_light_already_recorded = False

            # Update arrays with distance and next traffic light
            self.next_traffic_lights[i] = next_traffic_light
            self.distances_to_next_traffic_lights[i] = float(
                distance_idx) / self.points_per_meter

        # Since we search for traffic lights up to 5m away, we have to shift the arrays
        self.distances_to_next_traffic_lights = np.concatenate(
            [self.distances_to_next_traffic_lights[:-40], 40 * [np.inf]])
        self.next_traffic_lights = self.next_traffic_lights[:-40] + (40 * [
            None])

    def compute_distances_to_stop_signs(self, carla_world, carla_map):
        """
            Compute the distance to the next stop sign from each individual route location.
            We use the official implementation that is used to test whether we ran a stop sign
            The logic is copied from the class RunningStopTest in
            scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria

            Args:
                carla_world: Carla world instance.
                carla_map: Carla map instance.
            """

        def point_inside_boundingbox(point, bb_center, bb_extent, multiplier=1.2):
            """Checks whether or not a point is inside a bounding box."""

            A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x,
                               bb_center.y - multiplier * bb_extent.y)
            B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x,
                               bb_center.y - multiplier * bb_extent.y)
            D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x,
                               bb_center.y + multiplier * bb_extent.y)
            M = carla.Vector2D(point.x, point.y)

            AB = B - A
            AD = D - A
            AM = M - A
            am_ab = AM.x * AB.x + AM.y * AB.y
            ab_ab = AB.x * AB.x + AB.y * AB.y
            am_ad = AM.x * AD.x + AM.y * AD.y
            ad_ad = AD.x * AD.x + AD.y * AD.y

            return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad  # pylint: disable=chained-comparison

        def is_actor_affected_by_stop(wp_list, stop_extent, stop_location):
            """
                  Check if the given actor is affected by the stop.
                  Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
                  """

            # Quick distance test
            actor_location = wp_list[0].transform.location
            if stop_location.distance(actor_location) > 4.0:
                return False

            # Check if the any of the actor wps is inside the stop's bounding box.
            # Using more than one waypoint removes issues with small trigger volumes and backwards movement
            for actor_wp in wp_list:
                if point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
                    return True

            return False

        def _scan_for_stop_sign(list_stop_signs, list_stop_signs_extent, wp_list, stop_locations):
            """Check which stop sign affects the actor."""
            for (stop, stop_extent, stop_location) in zip(list_stop_signs, list_stop_signs_extent, stop_locations):
                if is_actor_affected_by_stop(wp_list, stop_extent, stop_location):
                    return stop

            return None

        def _get_waypoints(start_loc, carla_map):
            """Returns a list of waypoints starting from the ego location and a set amount forward"""
            wp_list = []
            steps = int(4.0 / 0.5)

            # Add the actor location
            wp = carla_map.get_waypoint(start_loc)
            wp_list.append(wp)

            # And its forward waypoints
            next_wp = wp
            for _ in range(steps):
                next_wps = next_wp.next(0.5)
                if not next_wps:
                    break
                next_wp = next_wps[0]
                wp_list.append(next_wp)

            return wp_list

        # Initialize arrays to store distances and next stop signs
        self.distances_to_next_stop_signs = np.full(
            self.route_points.shape[0], np.inf, dtype=np.float32)
        self.next_stop_signs = [None] * self.route_points.shape[0]

        # Get list of all stop signs
        list_stop_signs = carla_world.get_actors().filter("*traffic.stop*")

        next_stop_signs = None
        distance_idx = np.inf

        if list_stop_signs:
            list_stop_signs_extent = [
                x.trigger_volume.extent for x in list_stop_signs]

            # Adjust minimum extent for stop signs. That is necessary, since some stop signs are only 2cm thick
            # and because we use waypoints 50 cm apart it's likely we would miss it
            for extent in list_stop_signs_extent:
                extent.x = max(extent.x, 1)
                extent.y = max(extent.y, 1)

            stop_locations = [stop.get_transform().transform(
                stop.trigger_volume.location) for stop in list_stop_signs]
            stop_locations_np = np.array(
                [[x.x, x.y, x.z] for x in stop_locations])

            for i in range(self.route_points.shape[0]):
                loc = self.route_points[i]
                stop_sign = None

                # Quick distance check to safe computation later
                if np.linalg.norm(loc[None] - stop_locations_np, axis=1).min() < 4:
                    start_loc = carla.Location(x=loc[0], y=loc[1], z=loc[2])
                    check_wps = _get_waypoints(start_loc, carla_map)
                    stop_sign = _scan_for_stop_sign(
                        list_stop_signs, list_stop_signs_extent, check_wps, stop_locations)
                self.next_stop_signs[i] = stop_sign

            # Compute distances to next stop signs
            for i in range(self.distances_to_next_stop_signs.shape[0] - 1, -1, -1):
                if self.next_stop_signs[i] is not None:
                    next_stop_signs = self.next_stop_signs[i]
                    distance_idx = 0
                else:
                    distance_idx += 1

                self.next_stop_signs[i] = next_stop_signs
                self.distances_to_next_stop_signs[i] = float(
                    distance_idx) / self.points_per_meter

    def compute_leading_vehicles(self, list_vehicles, ego_vehicle_id):
        """
            Computes the IDs of vehicles leading ahead of the ego vehicle.

            Args:
                list_vehicles (list): List of all vehicles.
                ego_vehicle_id (int): ID of the ego vehicle.

            Returns:
                list: IDs of vehicles leading ahead of the ego vehicle.
            """
        # Get IDs of all vehicles except the ego vehicle
        vehicle_ids = np.array(
            [vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])

        # Check if there are vehicles and the route index is not at the end
        if len(vehicle_ids) and self.route_index != self.route_points.shape[0]:
            max_distance = self.leading_vehicles_maximum_detection_radius

            vehicle_yaws = np.array(
                [vehicle.get_transform().rotation.yaw for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = [vehicle.get_location(
            ) for vehicle in list_vehicles if vehicle.id != ego_vehicle_id]
            vehicle_locations = np.array(
                [[loc.x, loc.y, loc.z] for loc in vehicle_locations])

            # Compute leading vehicles up to 80m ahead
            # Computes if vehicle is leading ahead of the ego vehicle and its orientation is closer than
            # 35 degrees to the road
            # Both is necessary to ensure it is leading ahead of the ego vehicle and is not only crossing
            # its future path
            distances = vehicle_locations[:, None, :2] - self.route_points[None, self.route_index:self.route_index +
                                                                                                  max_distance, :2][:,
                                                         ::self.points_per_meter, :]
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[self.route_index:
                                                   self.route_index + max_distance][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(
                yaw_differences, 360 - yaw_differences)

            # Define the maximum distance and yaw difference thresholds
            max_distance = self.leading_vehicles_max_route_distance
            max_yaw_difference = self.leading_vehicles_max_route_angle_distance

            # Usually the road is 3.5 m wide, but in case of ParkingCrossingPedestrian it's less
            leading_vehicle_ids = vehicle_ids[(distances < max_distance) & (
                    yaw_differences < max_yaw_difference)]

            return leading_vehicle_ids.tolist()
        else:
            return []

    def compute_trailing_vehicles(self, list_vehicles, ego_vehicle_id):
        """
            Computes the IDs of vehicles trailing behind the ego vehicle.

            Args:
                list_vehicles (list): List of all vehicles.
                ego_vehicle_id (int): ID of the ego vehicle.

            Returns:
                list: IDs of vehicles trailing behind the ego vehicle
            """
        # Get IDs of all vehicles except the ego vehicle
        vehicle_ids = np.array(
            [vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])

        # Maximum distance of vehicles to ego's route
        max_distance = self.trailing_vehicles_max_route_distance

        # Check if there was a lane change in the past
        max_distance_lane_change = self.max_distance_lane_change_trailing_vehicles
        for i in range(max(0, self.route_index - max_distance_lane_change), self.route_index):
            if self.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                max_distance = self.trailing_vehicles_max_route_distance_lane_change
                break

        # Check if there are vehicles and the route index is not at the beginning
        if len(vehicle_ids) and self.route_index != 0:
            # Get yaw angles and locations of non-ego vehicles
            vehicle_yaws = np.array(
                [vehicle.get_transform().rotation.yaw for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = [vehicle.get_location(
            ) for vehicle in list_vehicles if vehicle.id != ego_vehicle_id]
            vehicle_locations = np.array(
                [[loc.x, loc.y, loc.z] for loc in vehicle_locations])

            max_distance_trailing_vehicles = self.tailing_vehicles_maximum_detection_radius
            # Computes if vehicle is behind ego vehicle and its orientation is closer than 30 degrees to the road
            # Both is necessary to ensure it is trailing the ego vehicle and is not only crossing its previous path
            from_idx = max(0, self.route_index -
                           max_distance_trailing_vehicles)
            distances = vehicle_locations[:, None, :2] - self.route_points[
                                                         None, from_idx:self.route_index, :2][:,
                                                         ::self.points_per_meter, :]
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[from_idx:
                                                   self.route_index][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(
                yaw_differences, 360 - yaw_differences)
            vehicles_behind_ids = vehicle_ids[(
                                                      distances < max_distance) & (yaw_differences < 30)]

            return vehicles_behind_ids.tolist()
        else:
            return []

    def world_to_grid(self, p):
        gx = int((p[0] - self.origin[0]) / self.grid_resolution)
        gy = int((p[1] - self.origin[1]) / self.grid_resolution)
        return gx, gy

    def get_future_a_star_path(self, ego_position):
        points_per_meter = self.points_per_meter
        route_points = self.route_points
        current_index = self.route_index
        max_points = int(points_per_meter * self.local_map_max_distance)
        total_points = route_points.shape[0]
        to_index = min(current_index + max_points, total_points)

        future_points_world = route_points[current_index:to_index]
        a_star_path = [(p[0], p[1]) for p in future_points_world]

        self.origin = (
            ego_position[0] - (self.grid_width * self.grid_resolution) / 2.0,
            ego_position[1] - (self.grid_height * self.grid_resolution) / 2.0
        )

        unique_grid_path = []
        last_point = None
        valid_end_index = current_index  

        for idx, p in enumerate(a_star_path):
            gx, gy = self.world_to_grid(p)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                current_point = (gx, gy)
                if current_point != last_point:
                    unique_grid_path.append(current_point)
                    last_point = current_point
                valid_end_index = current_index + idx  
            else:
                break
        return unique_grid_path, valid_end_index

    def initialize_dstar_lite(self, ego_position, unique_grid_path):

        if len(unique_grid_path) < 2:
            print("[D*] Not enough future points, skipping replan.")
            return False

        self.dstar_lite = DStarLite(self.grid_width, self.grid_height)
        start_node = Node(*unique_grid_path[0])
        goal_node = Node(*unique_grid_path[-1])

        self.dstar_lite.initial_start = start_node
        self.dstar_lite.initialize(start_node, goal_node)
        self.dstar_lite.inject_a_star_path(unique_grid_path)
        return True

    def insert_new_path(self, new_path, valid_end_index):
        """
        insert a new path into the route, ensuring smooth transitions and lane changes.
        """
        insert_start = self.route_index
        insert_end = valid_end_index

        if len(new_path) >= 2:
            heading_vec = (new_path[-1][0] - new_path[0][0], new_path[-1][1] - new_path[0][1])
            heading_norm = math.hypot(*heading_vec)
            if heading_norm > 1e-6:
                heading_vec = (heading_vec[0] / heading_norm, heading_vec[1] / heading_norm)
            else:
                heading_vec = (1.0, 0.0)
            lateral_vec = (-heading_vec[1], heading_vec[0])
        else:
            heading_vec = (1.0, 0.0)
            lateral_vec = (0.0, 1.0)

        smooth_new_path = []
        smooth_new_cmds = []
        i = 0
        back_steps = 2
        forward_steps = 2

        while i < len(new_path):
            if i > 0:
                prev = new_path[i - 1]
                curr = new_path[i]

                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]

                lateral_shift = dx * lateral_vec[0] + dy * lateral_vec[1]

                if abs(lateral_shift) > 0.5: 
                    start_idx = max(0, i - back_steps)
                    end_idx = min(len(new_path) - 1, i + forward_steps)

                    start_point = new_path[start_idx]
                    end_point = new_path[end_idx]

                    interp_points = np.linspace(start_point, end_point, end_idx - start_idx + 1)
                    smooth_new_path = smooth_new_path[:start_idx]
                    smooth_new_cmds = smooth_new_cmds[:start_idx]

                    for j in range(len(interp_points)):
                        smooth_new_path.append(interp_points[j])
                        smooth_new_cmds.append(RoadOption.CHANGELANELEFT)
                    i = end_idx + 1 
                    continue
            smooth_new_path.append(new_path[i])
            smooth_new_cmds.append(RoadOption.LANEFOLLOW)
            i += 1
        smooth_new_path = np.array(smooth_new_path)
        smooth_new_cmds = np.array(smooth_new_cmds)

        route_points = np.vstack([
            self.route_points[:insert_start],
            smooth_new_path,
            self.route_points[insert_end:]
        ])

        cmds = np.concatenate([
            self.commands[:insert_start],
            smooth_new_cmds,
            self.commands[insert_end:]
        ])

        self.route_points, self.commands = self.smooth_and_supersample(route_points, cmds)
        self.original_route_points = np.copy(self.route_points)
        self.commands_orig = self.commands.copy()

        self.route_waypoints = []
        for route_loc in self.route_points:
            wp = self.carla_map.get_waypoint(carla.Location(
                x=route_loc[0], y=route_loc[1], z=route_loc[2]))
            self.route_waypoints.append(wp)
        self.compute_route_info(self.carla_world, self.carla_map)

    def get_closest_waypoint(self, point):
        location = carla.Location(x=point[0], y=point[1], z=0)
        waypoint = self.carla_map.get_waypoint(location)
        return waypoint

    def replan_with_dstar_lite(self, ego_position, static_bounding_boxes):
        unique_grid_path, valid_end_index = self.get_future_a_star_path(ego_position)

        success = self.initialize_dstar_lite(ego_position, unique_grid_path)
        if not success:
            return None

        obstacle_list = set()
        expand = 1.75
        for obj in static_bounding_boxes:
            bbox = obj["bbox"]
            center = bbox.location
            extent = bbox.extent

            min_x = center.x - extent.y - expand
            max_x = center.x + extent.y + expand
            min_y = center.y - extent.x - expand
            max_y = center.y + extent.x + expand

            gx_min, gy_min = self.world_to_grid((min_x, min_y))
            gx_max, gy_max = self.world_to_grid((max_x, max_y))

            for gx in range(gx_min, gx_max + 1):
                for gy in range(gy_min, gy_max + 1):
                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        obstacle_list.add((gx, gy))
        obstacle_list = list(obstacle_list)

        print(unique_grid_path, flush=True)
        print(obstacle_list, flush=True)

        new_path_nodes = self.dstar_lite.run_navigation(unique_grid_path, obstacle_list)

        if new_path_nodes is None or len(new_path_nodes) == 0:
            print("[D*] Failed to find a path.")
            return None

        def grid_to_world(gx, gy):
            wx = gx * self.grid_resolution + self.origin[0]
            wy = gy * self.grid_resolution + self.origin[1]
            wz = self.route_points[self.route_index][2]
            return [wx, wy, wz]

        new_path = np.array([
            grid_to_world(node.x, node.y)
            for idx, node in enumerate(new_path_nodes)
            if idx % 3 == 0 or idx == len(new_path_nodes) - 1
        ])

        self.insert_new_path(new_path, valid_end_index)

        return new_path

    def add_obstacles(self, obstacle_list):
        for ox, oy in obstacle_list:
            self.dstar_lite.obstacles.add((ox, oy))
        for ox, oy in obstacle_list:
            node = Node(ox, oy)
            self.dstar_lite.update_vertex(node)
            for s in self.dstar_lite.get_neighbours(node):
                self.dstar_lite.update_vertex(s)

