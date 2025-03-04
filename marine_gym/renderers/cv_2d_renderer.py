from typing import List
import cv2
import numpy as np
from pydantic.v1.utils import deep_update

from ..types import Observation
from ..abstracts import AbcRender
from ..utils import ProfilingMeta

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 120, 0)
CYAN = (0, 255, 255)


def rgba(color, alpha, background=np.array([255, 255, 255])):
    return tuple((1 - alpha) * background + alpha * np.array(color))


def angle_to_vec_X(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def angle_to_vec_Y_naval(angle):
    return np.array([np.sin(angle), np.cos(angle)])


def rotate_vector(vector: np.ndarray, angle: float):
    assert vector.shape == (2,)
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rot @ vector

def draw_rotation_vector(img, center, radius, start_angle, end_angle, angular_velocity, color, thickness):
    """
    Draws a rotational vector as a curved arrow.
    
    :param img: The image to draw on
    :param center: Tuple (x, y) for the center of the rotation
    :param radius: Radius of the arc
    :param start_angle: Starting angle of the arc (degrees)
    :param end_angle: Ending angle of the arc (degrees)
    :param color: Color of the arc and arrowhead
    :param thickness: Thickness of the arc
    """
    # Convert angles from degrees to radians
    angle_rad = np.radians(end_angle)

    # Compute the arc end point (arrow tip)
    arrow_tip = (
        int(center[0] + radius * np.cos(angle_rad)),
        int(center[1] + radius * np.sin(angle_rad))
    )

    arrowhead_size = 8  # Reduce arrowhead size

    # Determine the direction of the arrowhead based on angular velocity
    if angular_velocity < 0:  # Clockwise (positive angular velocity)
        tangent_angle = angle_rad + np.pi / 2  # Perpendicular to the arc
    else:  # Counterclockwise (negative angular velocity)
        tangent_angle = angle_rad - np.pi / 2  # Perpendicular to the arc in opposite direction

    # Compute the arrowhead points
    arrow_end1 = (
        int(arrow_tip[0] + arrowhead_size * np.cos(tangent_angle - np.pi / 4)),
        int(arrow_tip[1] + arrowhead_size * np.sin(tangent_angle - np.pi / 4))
    )
    arrow_end2 = (
        int(arrow_tip[0] + arrowhead_size * np.cos(tangent_angle + np.pi / 4)),
        int(arrow_tip[1] + arrowhead_size * np.sin(tangent_angle + np.pi / 4))
    )

    # Draw the arc
    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness, lineType=cv2.LINE_AA)

    if np.abs(angular_velocity) > 0.001:
        # Draw the arrowhead
        cv2.line(img, arrow_tip, arrow_end1, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, arrow_tip, arrow_end2, color, thickness, lineType=cv2.LINE_AA)



class RendererObservation(metaclass=ProfilingMeta):
    def __init__(self, obs: Observation):

        # heading angle
        self.theta_boat = obs["theta_boat"][0]
        self.dt_theta_boat = obs["dt_theta_boat"][0]
        self.dt_theta_boat_vec = np.abs(obs["dt_theta_boat"][0]) * \
            angle_to_vec_X(self.theta_boat +
                         np.sign(obs["dt_theta_boat"][0]) * np.pi / 2)  # is either -90 or 90 degrees
        
        # position
        self.p_boat = np.array([obs["p_boat"][0], obs["p_boat"][1]])
        self.dt_p_boat = rotate_vector(
            np.array([obs["dt_p_boat"][0], obs["dt_p_boat"][1]]),
            self.theta_boat)

        # Main engine
        self.rpm = obs["rpm_me"][0]

        # Rudder
        self.theta_rudder = obs["theta_rudder"][0]
        self.dt_rudder = np.abs(obs["dt_theta_rudder"][0]) * \
            - angle_to_vec_X(-np.pi/2 + self.theta_rudder + np.sign(obs["dt_theta_rudder"][0]) * np.pi / 2)  # is either -90 or 90 degrees   
 
        # Thrusters
        self.bow_thruster = angle_to_vec_X(-self.theta_boat)*obs["N_thrusters"][0]
        self.stern_thruster = angle_to_vec_X(-self.theta_boat)*obs["N_thrusters"][1]


        # self.dt_sail = np.abs(obs["dt_theta_sail"][0]) * \
        #     angle_to_vec_X(self.theta_sail
        #                  + np.sign(obs["dt_theta_sail"][0]) * np.pi / 2)  # is either -90 or 90 degrees



        # wind
        self.wind = -angle_to_vec_Y_naval(obs["wind"][0]) * obs["wind"][1]

        # current
        self.current = angle_to_vec_Y_naval(obs["current"][0]) * obs["current"][1]

        # wave
        self.wave = -angle_to_vec_Y_naval(obs["wave"][0]) * obs["wave"][1]


class CV2DRenderer(AbcRender):
    def __init__(self, size=650, padding=50, vector_scale=50, style={}):
        self.size = size
        self.padding = padding
        self.vector_scale = vector_scale
        self.map_bounds = None
        self.center = None

        self.style = {
            "background": WHITE,
            "border": {
                "color": rgba(BLACK, .2),
                "width": 1,
            },
            "boat": {
                "color": rgba(BLACK, .3),
                "spike_coef": 2,
                "size": 50,
                "width": 13.50,  # Adjust width scaling
                "length": 94.70,  # Adjust length scaling
                "extension": 13.50,  # Bow extension for pointy front
                "phi": np.deg2rad(90),
                "dt_p": {
                    "color": GREEN,
                    "width": 1,
                },
                "dt_theta": {
                    "color": RED,
                    "width": 1,
                },
                "center": {
                    "color": WHITE,
                    "radius": 2,
                }
            },
            "rpm": {
                "color": RED,
                "width": 1,
                "height": 10,
                "dt_rpm": {
                    "color": RED,
                    "width": 1,
                },
            },
            "rudder": {
                "color": rgba(BLACK, .3),
                "width": 2,
                "height": 10,
                "dt_theta": {
                    "color": RED,
                    "width": 1,
                },
            },
            "thrusters": {
                "color": rgba(RED, .3),
                "width": 1,
                "height": 10,
            },
            "sail": {
                "color": rgba(BLACK, .7),
                "width": 2,
                "height": 15,
                "dt_theta": {
                    "color": RED,
                    "width": 1,
                },
            },
            "wind": {
                "color": rgba(BLUE, .5),
                "width": 2,
            },
            "current": {
                "color": rgba(CYAN, .5),
                "width": 2,
            },
            "wave": {
                "color": rgba(BLACK, .5),
                "width": 2,
            },
        }
        self.style = deep_update(self.style, style)

    def _create_empty_img(self):
        bg = np.array(self.style["background"])
        img = bg[None, None, :] + np.zeros((self.size, self.size, 3))
        return img.astype(np.uint8)

    def _scale_to_fit_in_img(self, x):
        self.internal_scaling = (self.size - 2 * self.padding) / (self.map_bounds[1] - self.map_bounds[0]).max() 
        return x * self.internal_scaling

    def _translate_and_scale_to_fit_in_map(self, x):
        return self._scale_to_fit_in_img(x - self.map_bounds[0]) + self.padding

    def _transform_obs_to_fit_in_img(self, obs: RendererObservation):
        # translate and scale positions
        obs.p_boat = self._translate_and_scale_to_fit_in_map(obs.p_boat)

        # scale vectors
        obs.dt_p_boat = self._scale_to_fit_in_img(obs.dt_p_boat)
        obs.dt_theta_boat_vec = self._scale_to_fit_in_img(obs.dt_theta_boat_vec)
        obs.dt_rudder = self._scale_to_fit_in_img(obs.dt_rudder)
        # obs.dt_sail = self._scale_to_fit_in_img(obs.dt_sail)
        obs.wind = self._scale_to_fit_in_img(obs.wind)
        obs.current = self._scale_to_fit_in_img(obs.current)
        obs.wave = self._scale_to_fit_in_img(obs.wave)

        obs.bow_thruster = self._scale_to_fit_in_img(obs.bow_thruster)
        obs.stern_thruster = self._scale_to_fit_in_img(obs.stern_thruster)



    def _draw_borders(self, img: np.ndarray):
        borders = self._translate_and_scale_to_fit_in_map(
            self.map_bounds).astype(int)
        cv2.rectangle(img,
                      tuple(borders[0]),
                      tuple(borders[1]),
                      self.style["border"]["color"],
                      self.style["border"]["width"],
                      lineType=cv2.LINE_AA)

    def _draw_wind(self, img: np.ndarray, obs: RendererObservation):
        img_center = np.array([2*self.padding, 2*self.padding])  
        cv2.arrowedLine(img,
                        tuple(img_center.astype(int)),
                        tuple((img_center + obs.wind *
                              self.vector_scale).astype(int)),
                        self.style["wind"]["color"],
                        self.style["wind"]["width"],
                        tipLength=0.2,
                        line_type=cv2.LINE_AA)

    def _draw_current(self, img: np.ndarray, obs: RendererObservation):
        img_center = np.array([2*self.padding, 2*self.padding])  
        cv2.arrowedLine(img,
                        tuple(img_center.astype(int)),
                        tuple((img_center + obs.current *
                              self.vector_scale).astype(int)),
                        self.style["current"]["color"],
                        self.style["current"]["width"],
                        tipLength=0.2,
                        line_type=cv2.LINE_AA)
        
    def _draw_wave(self, img: np.ndarray, obs: RendererObservation):
        img_center = np.array([2*self.padding, 2*self.padding])  
        cv2.arrowedLine(img,
                        tuple(img_center.astype(int)),
                        tuple((img_center + obs.wave *
                              self.vector_scale).astype(int)),
                        self.style["wave"]["color"],
                        self.style["wave"]["width"],
                        tipLength=0.2,
                        line_type=cv2.LINE_AA)

    def _draw_sailboat(self, img: np.ndarray, obs: RendererObservation):
        boat_size = self.style["boat"]["size"]
        phi = self.style["boat"]["phi"]
        spike_coeff = self.style["boat"]["spike_coef"]
        sailboat_pts = np.array([
            [obs.p_boat + angle_to_vec_X(obs.theta_boat + phi) * boat_size],
            [obs.p_boat + angle_to_vec_X(obs.theta_boat +
                                       (np.pi - phi)) * boat_size],
            [obs.p_boat + angle_to_vec_X(obs.theta_boat +
                                       (np.pi + phi)) * boat_size],
            [obs.p_boat + angle_to_vec_X(obs.theta_boat - phi) * boat_size],
            [obs.p_boat + angle_to_vec_X(obs.theta_boat)
             * spike_coeff * boat_size]
        ], dtype=int)
        cv2.fillConvexPoly(img,
                           sailboat_pts,
                           self.style["boat"]["color"],
                           lineType=cv2.LINE_AA)
        
    
    # Define ship shape (vertices)
    def generate_ship_shape(self, width, length, bow, scale, x=0, y=0, theta=0):

        # Define rotation matrix function
        def rotation_matrix(angle):
            # rotate 90 degrees and flip the sign of the rotation angle to match the naval coordinate system of representation visually 
            # the naval coordinate system is x - points north, y - points east, positive rotation - clockwise
            # Note: this system is equivelant with the simply rotated  90 degrees classical x - points east, y - points north, positive rotation - counter-clockwise, but 
            # only mirrowed (look at it from within the paper/glass, etc)
            return np.array([[0.0, -1.0], [1.0,  0.0]]) @ np.array([[np.cos(-1*angle), -np.sin(-1*angle)], [np.sin(-1*angle), np.cos(-1*angle)]])


        ship_shape = np.array([[0, 0], [length*scale, 0], [(length + bow)*scale, width*scale/2.0], [length*scale, width*scale], [0, width*scale], [0, 0]]).T - np.array([[length*scale/2., width*scale/2.]]).T

        # Rotate ship shape
        rotated_ship_shape  = rotation_matrix(theta) @ ship_shape

        # Translate ship shape
        transformed_ship_shape = rotated_ship_shape + np.array([[x, y]]).T

        return transformed_ship_shape

    def _draw_boat(self, img: np.ndarray, obs: RendererObservation):
        
        # Define ship dimensions
        boat_width = self.style["boat"]["width"] # Adjust width scaling
        boat_length = self.style["boat"]["length"]  # Adjust length scaling
        bow_extension = self.style["boat"]["extension"]  # Bow extension for pointy front
        scale = self.internal_scaling  # Scaling factor

        # Generate ship shape
        ship_shape = self.generate_ship_shape(
            width=boat_width,
            length=boat_length,
            bow=bow_extension,
            scale=scale,
            x=obs.p_boat[1],
            y=obs.p_boat[0],
            theta=obs.theta_boat
        )

        # Convert to integer pixel coordinates
        ship_shape = ship_shape.T.astype(int)

        # Draw the ship shape using fillConvexPoly
        cv2.fillConvexPoly(img, ship_shape, self.style["boat"]["color"], lineType=cv2.LINE_AA)


    def _draw_sail(self, img: np.ndarray, obs: RendererObservation):
        sail_height = self.style["sail"]["height"]
        sail_start = obs.p_boat
        sail_end = sail_start + angle_to_vec_X(obs.theta_sail) * sail_height
        cv2.line(img,
                 tuple(sail_start.astype(int)),
                 tuple(sail_end.astype(int)),
                 self.style["sail"]["color"],
                 self.style["sail"]["width"],
                 lineType=cv2.LINE_AA)

    def _draw_rudder(self, img: np.ndarray, obs: RendererObservation):
        rudder_height = self.style["rudder"]["height"]
        boat_phi = self.style["boat"]["phi"]
        boat_size = self.style["boat"]["size"]

        back_of_boat = obs.p_boat[::-1] - angle_to_vec_Y_naval(obs.theta_boat) * boat_size*self.internal_scaling
        rudder_start = back_of_boat
        rudder_end = rudder_start - angle_to_vec_Y_naval(-(-obs.theta_boat + obs.theta_rudder)) * rudder_height
        
        # - angle_to_vec_X(-np.pi/2 + obs.theta_rudder) * rudder_height
        cv2.line(img,
                 tuple(rudder_start.astype(int)),
                 tuple(rudder_end.astype(int)),
                 self.style["rudder"]["color"],
                 self.style["rudder"]["width"],
                 lineType=cv2.LINE_AA)
        
    def _draw_me_rpm(self, img: np.ndarray, obs: RendererObservation):
        boat_size = self.style["boat"]["size"]
        # reverse the vector because marine coordinate system is flipped y-x
        propeller_start = obs.p_boat[::-1] - angle_to_vec_Y_naval(obs.theta_boat) * boat_size*self.internal_scaling/2
        propeller_end = propeller_start - angle_to_vec_Y_naval(obs.theta_boat) * obs.rpm  / 10
        cv2.arrowedLine(img,
                        tuple(propeller_start.astype(int))  ,
                        tuple(propeller_end.astype(int)),
                        self.style["rpm"]["color"],
                        self.style["rpm"]["width"],
                        tipLength=.2,
                        line_type=cv2.LINE_AA)
        
    
    def _draw_thrusters(self, img: np.ndarray, obs: RendererObservation):
        boat_size = self.style["boat"]["size"]

        # reverse the vector because marine coordinate system is flipped y-x
        bow_thruster_start = obs.p_boat[::-1] + angle_to_vec_Y_naval(obs.theta_boat) * 7*boat_size*self.internal_scaling/8.
        bow_thruster_end = bow_thruster_start + obs.bow_thruster * self.vector_scale / 60000 # manual scaling to be changed

        cv2.arrowedLine(img,
                        tuple(bow_thruster_start.astype(int)),
                        tuple(bow_thruster_end.astype(int)),
                        self.style["thrusters"]["color"],
                        self.style["thrusters"]["width"],
                        tipLength=.2,
                        line_type=cv2.LINE_AA)
        
        # reverse the vector because marine coordinate system is flipped y-x
        stern_thruster_start = obs.p_boat[::-1] - angle_to_vec_Y_naval(obs.theta_boat) * 7*boat_size*self.internal_scaling/8.
        stern_thruster_end = stern_thruster_start + obs.stern_thruster * self.vector_scale / 60000 # manual scaling to be changed
        cv2.arrowedLine(img,
                        tuple(stern_thruster_start.astype(int)),
                        tuple(stern_thruster_end.astype(int)),
                        self.style["thrusters"]["color"],
                        self.style["thrusters"]["width"],
                        tipLength=.2,
                        line_type=cv2.LINE_AA)
        
    def _draw_boat_pos_velocity(self, img: np.ndarray, obs: RendererObservation):
        dt_p_boat_start = obs.p_boat
        dt_p_boat_end = dt_p_boat_start + obs.dt_p_boat * self.vector_scale
        cv2.arrowedLine(img,
                        tuple(dt_p_boat_start.astype(int))[::-1],  # reverse the tuple because marine coordinate system is flipped y-x
                        tuple(dt_p_boat_end.astype(int))[::-1],    # reverse the tuple because marine coordinate system is flipped y-x
                        self.style["boat"]["dt_p"]["color"],
                        self.style["boat"]["dt_p"]["width"],
                        tipLength=.2,
                        line_type=cv2.LINE_AA)

    def _draw_boat_heading_velocity(self, img: np.ndarray, obs: RendererObservation):
        spike_coeff = self.style["boat"]["spike_coef"]
        boat_size = self.style["boat"]["size"]

        # reverse the vector because marine coordinate system is flipped y-x
        front_of_boat = obs.p_boat[::-1] + \
            angle_to_vec_Y_naval(obs.theta_boat) * spike_coeff * boat_size*self.internal_scaling
 
        dt_theta_boat_vec_start = front_of_boat
        dt_theta_boat_vec_end = dt_theta_boat_vec_start + obs.dt_theta_boat_vec * self.vector_scale
         
        radius = 10  # Size of arc
        start_angle = 0  # Starting angle of the arc
        end_angle = -np.rad2deg(obs.dt_theta_boat) * self.vector_scale * 10 # Ending angle of the arc
        angular_velocity = -np.rad2deg(obs.dt_theta_boat)
        color = (0, 200, 0)  # Green color
        thickness = 1  # Thickness of arc and lines
        draw_rotation_vector(img, 
                             tuple(obs.p_boat.astype(int))[::-1],   # reverse the tuple because marine coordinate system is flipped y-x
                             radius, 
                             start_angle, 
                             end_angle, 
                             angular_velocity,
                             color, 
                             thickness)


    def _draw_rudder_velocity(self, img: np.ndarray, obs: RendererObservation):
        boat_phi = self.style["boat"]["phi"]
        boat_size = self.style["boat"]["size"]
        rudder_height = self.style["rudder"]["height"]
        back_of_boat = obs.p_boat[::-1] - angle_to_vec_Y_naval(obs.theta_boat) * boat_size*self.internal_scaling
        rudder_end = back_of_boat - angle_to_vec_Y_naval(-obs.theta_rudder) * rudder_height
        dt_rudder_start = rudder_end
        dt_rudder_end = dt_rudder_start - obs.dt_rudder*200 # manual scaling to be changed
        cv2.arrowedLine(img,
                        tuple(dt_rudder_start.astype(int)),
                        tuple(dt_rudder_end.astype(int)),
                        self.style["rudder"]["dt_theta"]["color"],
                        self.style["rudder"]["dt_theta"]["width"],
                        tipLength=.2,
                        line_type=cv2.LINE_AA)

    def _draw_sail_velocity(self, img: np.ndarray, obs: RendererObservation):
        sail_height = self.style["sail"]["height"]
        dt_sail_start = obs.p_boat + \
            angle_to_vec_X(obs.theta_sail) * sail_height
        dt_sail_end = dt_sail_start + obs.dt_sail
        cv2.arrowedLine(img,
                        tuple(dt_sail_start.astype(int)),
                        tuple(dt_sail_end.astype(int)),
                        self.style["sail"]["dt_theta"]["color"],
                        self.style["sail"]["dt_theta"]["width"],
                        tipLength=.2,
                        line_type=cv2.LINE_AA)

    def _draw_boat_center(self, img: np.ndarray, obs: RendererObservation):
        cv2.circle(img,
                   tuple(obs.p_boat.astype(int)),
                   self.style["boat"]["center"]["radius"],
                   self.style["boat"]["center"]["color"],
                   -1)
        
    
    
    def draw_vessel_path(self, img, waypoints, color=(0, 0, 255), thickness=1):
        """
        Draws a path given by waypoints on an img and labels each waypoint with its index.
        
        :param img: The img on which to draw.
        :param waypoints: List of (x, y) tuples representing waypoints.
        :param color: Color of the path and waypoint markers (default green).
        :param thickness: Thickness of the path line.
        """
        
        height, width = img.shape[:2]  # Get image dimensions

        # Convert waypoints to integer pixel values
        waypoints = np.array(waypoints, dtype=np.int32)

        # Draw the path
        cv2.polylines(img, [waypoints], isClosed=False, color=color, thickness=thickness)

        # Draw waypoints and add indices
        for i, (x, y) in enumerate(waypoints):
            cv2.circle(img, (x, y), radius=5, color=color, thickness=-1)  # Mark waypoint
            # label = f"{i} ({x}, {y})"
            label = f"{i}"

            cv2.putText(img, label, (int(x) + 5, int(y) - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA, True)  # Add index with coordinates

        return img
    
    def _render_sailboat_obs(self, img, obs):
            
            # draw map
            self._draw_borders(img)
            self._draw_wind(img, obs)
            self._draw_current(img, obs)
            self._draw_boat(img, obs)
            self._draw_boat_heading_velocity(img, obs)
            self._draw_boat_pos_velocity(img, obs)
            self._draw_rudder(img, obs)
            self._draw_rudder_velocity(img, obs)
            self._draw_sail(img, obs)
            self._draw_sail_velocity(img, obs)
            self._draw_boat_center(img, obs)
           
    def _render_ship_obs(self, img, obs):
            
            # draw map
            self._draw_borders(img)

            if self.scaled_path is not None:
                self.draw_vessel_path(img, self.scaled_path)

            self._draw_boat_center(img, obs)

            self._draw_wind(img, obs)
            self._draw_current(img, obs)
            self._draw_wave(img, obs)
            self._draw_boat(img, obs)
            self._draw_boat_heading_velocity(img, obs)
            self._draw_boat_pos_velocity(img, obs)

            self._draw_me_rpm(img, obs)
            self._draw_rudder(img, obs)
            self._draw_rudder_velocity(img, obs)
            self._draw_thrusters(img, obs)
            
    def get_render_mode(self) -> str:
        return 'rgb_array'

    def get_render_modes(self) -> List[str]:
        return ['rgb_array']

    def setup(self, map_bounds, ref_path):
        self.map_bounds = map_bounds[:, 0:2]  # ignore z axis
        self.center = (self.map_bounds[0] + self.map_bounds[1]) / 2

        # show reference path
        self.scaled_path = None
        if ref_path is not None:
            self.scaled_path = self._translate_and_scale_to_fit_in_map(ref_path)
        
    def render(self, observation, draw_extra_fct=None):
        assert (self.map_bounds is not None
                and self.center is not None), "Please call setup() first."

        img = self._create_empty_img()

        # prepare observation
        obs = RendererObservation(observation)
        self._transform_obs_to_fit_in_img(obs)

        # draw extra stuff
        if draw_extra_fct is not None:
            draw_extra_fct(img, observation)

        # draw map
        self._render_ship_obs(img, obs)

        # flip vertically
        img = img[::-1, :, :]

        return img
