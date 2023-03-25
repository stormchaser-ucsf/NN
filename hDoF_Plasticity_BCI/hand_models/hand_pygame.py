# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:49:46 2023

@author: nikic
"""

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/hand_models')
import pygame
import math

pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set the width and height of the screen [width, height]
size = (700, 800)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Virtual Hand")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()


# Hand dimensions and initial position
HAND_WIDTH = 200
HAND_HEIGHT = 300
HAND_X = 250
HAND_Y = 100

# Finger dimensions
FINGER_WIDTH = 80
FINGER_HEIGHT = 150

# Set the initial angles for each joint of the hand
thumb_angle = 0
index_angle = 0
middle_angle = 0
ring_angle = 0
little_angle = 0

# Main game loop
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Set the background color
    screen.fill(WHITE)

    # Draw the hand
    pygame.draw.rect(screen, BLACK, [HAND_X, HAND_Y, HAND_WIDTH, HAND_HEIGHT])

    # Draw the thumb
    thumb_tip_x = HAND_X + HAND_WIDTH
    thumb_tip_y = HAND_Y + FINGER_HEIGHT
    thumb_base_x = thumb_tip_x - 50
    thumb_base_y = thumb_tip_y
    thumb_tip_rotated_x = math.cos(thumb_angle) * (thumb_tip_x - thumb_base_x) - math.sin(thumb_angle) * (thumb_tip_y - thumb_base_y) + thumb_base_x
    thumb_tip_rotated_y = math.sin(thumb_angle) * (thumb_tip_x - thumb_base_x) + math.cos(thumb_angle) * (thumb_tip_y - thumb_base_y) + thumb_base_y
    pygame.draw.line(screen, BLACK, [thumb_base_x, thumb_base_y], [thumb_tip_rotated_x, thumb_tip_rotated_y], 50)

    # Draw the index finger
    index_tip_x = HAND_X
    index_tip_y = HAND_Y + FINGER_HEIGHT
    index_base_x = index_tip_x + FINGER_WIDTH
    index_base_y = index_tip_y
    index_tip_rotated_x = math.cos(index_angle) * (index_tip_x - index_base_x) - math.sin(index_angle) * (index_tip_y - index_base_y) + index_base_x
    index_tip_rotated_y = math.sin(index_angle) * (index_tip_x - index_base_x) + math.cos(index_angle) * (index_tip_y - index_base_y) + index_base_y
    pygame.draw.line(screen, BLACK, [index_base_x, index_base_y], [index_tip_rotated_x, index_tip_rotated_y], 50)

    # Draw the middle finger
    middle_tip_x = HAND_X
    middle_tip_y = HAND_Y + FINGER_HEIGHT * 2
    middle_base_x = middle_tip_x + FINGER_WIDTH
    middle_base_y = middle_tip_y
    middle_tip_rotated_x = math.cos(middle_angle) * (middle_tip_x - middle_base_x) - math.sin(middle_angle) * (middle_tip_y - middle_base_y) + middle_base_x
    middle_tip_rotated_y = math.sin(middle_angle) * (middle_tip_x - middle_base_x) + math.cos(middle_angle) * (middle_tip_y - middle_base_y) + middle_base_y    
    pygame.draw.line(screen, BLACK, [middle_base_x, middle_base_y], [middle_tip_rotated_x, middle_tip_rotated_y], 50)
    
    # Draw the ring finger
    ring_tip_x = HAND_X
    ring_tip_y = HAND_Y + FINGER_HEIGHT * 3
    ring_base_x = ring_tip_x + FINGER_WIDTH
    ring_base_y = ring_tip_y
    ring_tip_rotated_x = math.cos(ring_angle) * (ring_tip_x - ring_base_x) - math.sin(ring_angle) * (ring_tip_y - ring_base_y) + ring_base_x
    ring_tip_rotated_y = math.sin(ring_angle) * (ring_tip_x - ring_base_x) + math.cos(ring_angle) * (ring_tip_y - ring_base_y) + ring_base_y
    pygame.draw.line(screen, BLACK, [ring_base_x, ring_base_y], [ring_tip_rotated_x, ring_tip_rotated_y], 50)
    
    # Draw the little finger
    little_tip_x = HAND_X
    little_tip_y = HAND_Y + FINGER_HEIGHT * 4
    little_base_x = little_tip_x + FINGER_WIDTH
    little_base_y = little_tip_y
    little_tip_rotated_x = math.cos(little_angle) * (little_tip_x - little_base_x) - math.sin(little_angle) * (little_tip_y - little_base_y) + little_base_x
    little_tip_rotated_y = math.sin(little_angle) * (little_tip_x - little_base_x) + math.cos(little_angle) * (little_tip_y - little_base_y) + little_base_y
    pygame.draw.line(screen, BLACK, [little_base_x, little_base_y], [little_tip_rotated_x, little_tip_rotated_y], 50)
    
    # Update the joint angles based on user input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        thumb_angle += 0.05
    elif keys[pygame.K_a]:
        thumb_angle -= 0.05
    if keys[pygame.K_w]:
        index_angle += 0.05
    elif keys[pygame.K_s]:
        index_angle -= 0.05
    if keys[pygame.K_e]:
        middle_angle += 0.05
    elif keys[pygame.K_d]:
        middle_angle -= 0.05
    if keys[pygame.K_r]:
        ring_angle += 0.05
    elif keys[pygame.K_f]:
        ring_angle -= 0.05
    if keys[pygame.K_t]:
        little_angle += 0.05
    elif keys[pygame.K_g]:
        little_angle -= 0.05
    
    # Flip the display
    pygame.display.flip()
    
    # Limit to 60 frames per second
    clock.tick(60)

    
