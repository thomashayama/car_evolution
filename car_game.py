import pygame
from line_seg import Line_seg
from vector import Vector2d
from car import Car
import math
import roads

pygame.init()

display_width = 800
display_height = 600

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Car')

gray = (100, 100, 100)
white = (255, 255, 255)
red = (255, 100, 100)
green = (100, 255, 100)

clock = pygame.time.Clock()
crashed = False


key_pressed = {
    'right': False,
    'left': False,
    'up': False,
    'down': False,
    'space': False
}

x = (display_width * 0.45)
y = (display_height * 0.5)
p1 = Car(Vector2d(100, 100), 40, 20)
speed = 200
curr_goal = 0

road = roads.road2.data
goals = roads.road2.goals

while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                key_pressed['left'] = True
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                key_pressed['right'] = True
            if event.key == pygame.K_w:
                key_pressed['up'] = True
            if event.key == pygame.K_s:
                key_pressed['down'] = True
            if event.key == pygame.K_SPACE:
                key_pressed['space'] = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                key_pressed['left'] = False
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                key_pressed['right'] = False
            if event.key == pygame.K_w:
                key_pressed['up'] = False
            if event.key == pygame.K_s:
                key_pressed['down'] = False
            if event.key == pygame.K_SPACE:
                key_pressed['space'] = False

    if key_pressed['left']:
        p1.ang_vel = -1
    elif key_pressed['right']:
        p1.ang_vel = 1
    else:
        p1.ang_vel = 0
    if key_pressed['up']:
        #p1.vel.x = speed * math.cos(p1.angle/180*math.pi)
        #p1.vel.y = speed * math.sin(p1.angle/180*math.pi)
        p1.acc.x = 200 * math.cos(p1.angle / 180 * math.pi)
        p1.acc.y = 200 * math.sin(p1.angle / 180 * math.pi)
    elif key_pressed['down']:
        #p1.vel.x = -speed * math.cos(p1.angle/180*math.pi)
        #p1.vel.y = -speed * math.sin(p1.angle/180*math.pi)
        p1.acc.x = -200 * math.cos(p1.angle / 180 * math.pi)
        p1.acc.y = -200 * math.sin(p1.angle / 180 * math.pi)
    else:
        p1.acc.x = 0
        p1.acc.y = 0


    gameDisplay.fill(gray)
    for seg in goals:
        seg.show(gameDisplay, green)

    if p1.tick(clock.get_time()/1000, gameDisplay, road, goals[curr_goal % len(goals)], show=True, color=red):
        print('goal')
        curr_goal += 1

    if p1.is_done:
        print('hit wall')
        p1.is_done = False

    for seg in road:
        seg.show(gameDisplay, white)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()