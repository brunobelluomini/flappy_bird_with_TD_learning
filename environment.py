import numpy as np
from itertools import cycle
import pickle
import random
import sys

import pygame
from pygame.locals import *


def create_uniform_grid(width, height):
    """
    Create grids of 10x10 pixels
    """
    grid_size = (10, 10)
    num_bins_horizontal = int(round(width / grid_size[0]))
    num_bins_vertical = int(round(height / grid_size[1]))
    bins = (num_bins_horizontal, num_bins_vertical)
    low = [0, 0]
    high = [width, height]
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

    return grid


def map_position_tile(position, grid):
    """
    Map some position acording to the discretized game space
    Input: position (list): List with position [x, y]
    Output: grid (list): Tile which matches the given position for a given grid
    """
    return list(int(np.digitize(p, g)) for p, g in zip(position, grid))


class FlappyEnvironment:


    def __init__(self):
        # General game settings
        self.action_space = 2
        self.width = 288
        self.height = 512
        self.base_y = self.height * 0.79
        self.game_grid = create_uniform_grid(self.width, self.height)
        self.game_score = 0
        self.has_scored = False
        self.loop_iter = 0

        # Bird (player) settings
        self.bird_width = 34
        self.bird_height = 24
        self.bird_pos_x = 57
        self.bird_pos_y = 244
        self.bird_index_gen = cycle([0, 1, 2, 1])
        self.bird_index = 0
        self.bird_vel_y = -9
        self.bird_max_vel_y = 10
        self.bird_min_vel_y = -9
        self.bird_acc_y = 1    
        self.bird_flap_acc = -9   
        self.bird_flapped = False

        # Pipe settings
        self.pipe_vel_x = -4
        self.pipe_gap_size = 100
        self.pipe_width = 52
        self.pipe_height = 320


        with open('hitmasks_data.pkl', 'rb') as self.hitmasks_data_input:
            self.hitmasks = pickle.load(self.hitmasks_data_input)

        # Get 2 new pipes to add to upper_pipes and lower_pipes list
        new_pipe_1 = self.get_random_pipe()
        new_pipe_2 = self.get_random_pipe()

        self.upper_pipes = [
            {'x': self.width + 200, 'y': new_pipe_1[0]['y']},
            {'x': self.width + 200 + (self.width / 2), 'y': new_pipe_2[0]['y']}
        ]

        self.lower_pipes = [
            {'x': self.width + 200, 'y': new_pipe_1[1]['y']},
            {'x': self.width + 200 + (self.width / 2), 'y': new_pipe_2[1]['y']}
        ]

    
    def step(self, action):
        self.has_scored = False

        if action == 1:
            if self.bird_pos_y > -2 * self.bird_height:
                self.bird_vel_y = self.bird_flap_acc
                self.bird_flapped = True
        
        collision_check = self.check_collision()

        if collision_check:
            done = True
        else:
            done = False
        
        # Update game score
        bird_mid_pos = self.bird_pos_x + self.bird_width / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + self.pipe_width / 2

            if pipe_mid_pos <= bird_mid_pos < pipe_mid_pos + 4:
                self.has_scored = True
                self.game_score += 1
        
        # Player index change
        if (self.loop_iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_gen)
        
        self.loop_iter = (self.loop_iter + 1) % 30

        # Move the bird
        if self.bird_vel_y < self.bird_max_vel_y and not self.bird_flapped:
            self.bird_vel_y += self.bird_acc_y
        
        if self.bird_flapped:
            self.bird_flapped = False
        
        self.bird_pos_y += min(self.bird_vel_y, self.base_y - self.bird_pos_y - self.bird_height)

        # Move pipes to left
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe['x'] += self.pipe_vel_x
            lower_pipe['x'] += self.pipe_vel_x

        # Add new pipes when first pipe is about to touch left of screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self.get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # Remove the first pipe if it's out of the screen
        if self.upper_pipes[0]['x'] < -self.pipe_width:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # Get the next state
        next_state = self.get_state()
        reward = self.get_reward(collision_check, next_state)

        return next_state, reward, done


    def check_collision(self):
        # if player crashes into ground
        if (self.bird_pos_y + self.bird_height >= self.base_y - 1 ) or (self.bird_pos_y + self.bird_height <= 0):
            return True

        else:
            bird_rectangle = pygame.Rect(
                self.bird_pos_x,
                self.bird_pos_y,
                self.bird_width,
                self.bird_height
            )
            for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
                # Make upper and lower pipe rectangles
                upper_pipe_rectangle = pygame.Rect(upper_pipe['x'], upper_pipe['y'], self.pipe_width, self.pipe_height)
                lower_pipe_rectangle = pygame.Rect(lower_pipe['x'], lower_pipe['y'], self.pipe_width, self.pipe_height)

                bird_hitmask = self.hitmasks['player'][self.bird_index]
                upper_pipe_hitmask = self.hitmasks['pipe'][0]
                lower_pipe_hitmask = self.hitmasks['pipe'][1]

                # if bird collided with upper_pipe_hitmask or lower_pipe_hitmask
                upper_collision = self.check_pixel_collision(
                    bird_rectangle, 
                    upper_pipe_rectangle, 
                    bird_hitmask, 
                    upper_pipe_hitmask
                )
                
                lower_collision = self.check_pixel_collision(
                    bird_rectangle, 
                    lower_pipe_rectangle, 
                    bird_hitmask, 
                    lower_pipe_hitmask
                )

                if upper_collision or lower_collision:
                    return True

        return False

    
    def check_pixel_collision(self, bird_rect, pipe_rect, bird_hitmask, pipe_hitmask):
        rectangle = bird_rect.clip(pipe_rect)

        if rectangle.width == 0 or rectangle.height == 0:
            return False

        x1, y1 = rectangle.x - bird_rect.x, rectangle.y - bird_rect.y
        x2, y2 = rectangle.x - pipe_rect.x, rectangle.y - pipe_rect.y

        for x in range(rectangle.width):
            for y in range(rectangle.height):
                if bird_hitmask[x1 + x][y1 + y] and pipe_hitmask[x2 + x][y2 + y]:
                    return True
        
        return False


    def get_random_pipe(self):
        gap_y = random.randrange(0, int(self.base_y * 0.6 - self.pipe_gap_size))
        gap_y += int(self.base_y * 0.2)
        pipe_x = self.width + 10

        upper_pipe = {'x': pipe_x, 'y': gap_y - self.pipe_height}
        lower_pipe = {'x': pipe_x, 'y': gap_y + self.pipe_gap_size}

        return [upper_pipe, lower_pipe]


    def get_reward(self, collision, next_state):
        vertical_pos_diff = int(next_state.split('_')[1])

        if collision:
            return -10000

        elif self.has_scored:
            return 1000

        elif abs(vertical_pos_diff) >= 2:
            return -10        

        else:
            return 10


    def get_state(self):
        if self.lower_pipes[0]['x'] - self.bird_pos_x > -30:
            pipe = self.lower_pipes[0]
        else:
            pipe = self.lower_pipes[1]

        bird_tile_pos = map_position_tile([self.bird_pos_x, self.bird_pos_y], self.game_grid)
        mid_pipe_gap_tile_pos = map_position_tile([pipe['x'], pipe['y'] + self.pipe_gap_size / 2], self.game_grid)
        pos_difference = np.subtract(mid_pipe_gap_tile_pos, bird_tile_pos)
        dist_x = pos_difference[0]
        dist_y = pos_difference[1]
        vel_y = self.bird_vel_y

        return f"{dist_x}_{dist_y}_{vel_y}"