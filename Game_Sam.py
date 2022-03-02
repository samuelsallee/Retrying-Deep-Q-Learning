import pygame
from pygame.locals import *
import math

pygame.init()

SCREEN_WIDTH: int = 1050
SCREEN_HEIGHT: int = 900
RENDER = False

if RENDER:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), RESIZABLE)

x = 300
y = 400


class Car:
    def __init__(self):
        self.x = 100
        self.y = 400
        self.img = pygame.transform.scale(
            pygame.image.load('toppng.com-aston-martin-one77-01-top-down-car-sprite-480x920.png'), (30, 30))
        self.imgcpy = pygame.transform.scale(
            pygame.image.load('toppng.com-aston-martin-one77-01-top-down-car-sprite-480x920.png'), (30, 30))

        self.outline = [
            ((self.x, self.y), (self.x, self.y + 30)),
            ((self.x, self.y + 30), (self.x + 15, self.y + 30)),
            ((self.x + 15, self.y + 30), (self.x + 15, self.y)),
            ((self.x + 15, self.y), (self.x, self.y))
        ]

        self.rotation = 0-math.pi/64
        self.speed = 0
        self.center2 = (self.x + 15, self.y + 15)

        self.center = (self.center2[0] + 15, self.center2[1] + 15)

        self.vision = [
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi), self.center[1] + 1000 * math.cos(-self.rotation + math.pi))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi/2), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi/2))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi/2), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi/2))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi*3/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi*3/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi*3/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi*3/4)))
            ]

    def update(self):
        self.shape = ((self.x, self.y), (15, 30))
        self.center = (self.center2[0] + 15, self.center2[1] + 15)
        self.vision = [
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi), self.center[1] + 1000 * math.cos(-self.rotation + math.pi))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi/2), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi/2))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi/2), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi/2))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi*3/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi*3/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi*3/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi*3/4)))
            ]

    def reset(self):
        self.x = 100
        self.y = 400
        self.img = pygame.transform.scale(
            pygame.image.load('toppng.com-aston-martin-one77-01-top-down-car-sprite-480x920.png'), (30, 30))
        self.imgcpy = pygame.transform.scale(
            pygame.image.load('toppng.com-aston-martin-one77-01-top-down-car-sprite-480x920.png'), (30, 30))

        self.outline = [
            ((self.x, self.y), (self.x, self.y + 30)),
            ((self.x, self.y + 30), (self.x + 15, self.y + 30)),
            ((self.x + 15, self.y + 30), (self.x + 15, self.y)),
            ((self.x + 15, self.y), (self.x, self.y))
        ]

        self.rotation = 0-math.pi/64
        self.speed = 0
        self.center2 = (self.x + 15, self.y + 15)

        self.center = (self.center2[0] + 15, self.center2[1] + 15)

        self.vision = [
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi), self.center[1] + 1000 * math.cos(-self.rotation + math.pi))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi/2), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi/2))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi/2), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi/2))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi + math.pi*3/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi + math.pi*3/4))),
            (self.center,
             (self.center[0] + 1000 * math.sin(-self.rotation + math.pi - math.pi*3/4), self.center[1] + 1000 * math.cos(-self.rotation + math.pi - math.pi*3/4)))
            ]


car = Car()
quit = False

lines = [
    ((50, 600), (50, 200)),
    ((50, 200), (200, 50)),
    ((200, 50), (850, 50)),
    ((850, 50), (1000, 200)),
    ((1000, 200), (1000, 600)),
    ((1000, 600), (850, 750)),
    ((850, 750), (200, 750)),
    ((200, 750), (50, 600)),

    ((150, 575), (150, 225)),
    ((150, 225), (225, 150)),
    ((225, 150), (825, 150)),
    ((825, 150), (900, 225)),
    ((900, 225), (900, 575)),
    ((900, 575), (825, 650)),
    ((825, 650), (225, 650)),
    ((225, 650), (150, 575))
]

checkpoints = [
    ((50,350), (150,350)),
    ((50, 300), (150, 300)),
    ((50, 250), (150, 250)),

    ((85, 160), (150, 225)),
    ((85 + 35, 160 - 35), (150 + 35, 225 - 35)),
    ((85 + 35 + 35, 160 - 35 - 35), (150 + 35 + 35, 225 - 35 - 35)),

    ((250, 50), (250, 150)),
    ((300, 50), (300, 150)),
    ((350, 50), (350, 150)),
    ((400, 50), (400, 150)),
    ((450, 50), (450, 150)),
    ((500, 50), (500, 150)),
    ((550, 50), (550, 150)),
    ((600, 50), (600, 150)),
    ((650, 50), (650, 150)),
    ((700, 50), (700, 150)),
    ((750, 50), (750, 150)),
    ((800, 50), (800, 150)),

    ((825, 150), (885, 90)),
    ((825 + 35, 150 + 35), (885 + 35, 90 + 35)),
    ((825 + 35 + 35, 150 + 35 + 35), (885 + 35 + 35, 90 + 35 + 35)),

    ((900, 250), (1000, 250)),
    ((900, 300), (1000, 300)),
    ((900, 350), (1000, 350)),
    ((900, 400), (1000, 400)),
    ((900, 450), (1000, 450)),
    ((900, 500), (1000, 500)),
    ((900, 550), (1000, 550)),

    ((900, 575), (960, 635)),
    ((900-35, 575+35), (960-35, 635+35)),
    ((900 - 35 - 35, 575 + 35 + 35), (960 - 35 - 35, 635 + 35 + 35)),

    ((800, 650), (800, 750)),
    ((750, 650), (750, 750)),
    ((700, 650), (700, 750)),
    ((650, 650), (650, 750)),
    ((600, 650), (600, 750)),
    ((550, 650), (550, 750)),
    ((500, 650), (500, 750)),
    ((450, 650), (450, 750)),
    ((400, 650), (400, 750)),
    ((350, 650), (350, 750)),
    ((300, 650), (300, 750)),
    ((250, 650), (250, 750)),

    ((225, 650), (165, 710)),
    ((225-35, 650-35), (165-35, 710-35)),
    ((225-35-35, 650-35-35), (165-35-35, 710-35-35)),

    ((50, 550), (150, 550)),
    ((50, 500), (150, 500)),
    ((50, 450), (150, 450)),
    ((50, 400), (150, 400))
]

state = []

next_checkpoint = 0
last_checkpoint = -2
event = 0


def line_intersection(line1, line2):
    """ returns a (x, y) tuple or None if there is no intersection """
    Ax1 = line1[0][0]
    Ay1 = line1[0][1]
    Ax2 = line1[1][0]
    Ay2 = line1[1][1]
    Bx1 = line2[0][0]
    By1 = line2[0][1]
    Bx2 = line2[1][0]
    By2 = line2[1][1]
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    point = (-5000, -5000)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return point
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return point
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    point = (x,y)
    return point


crashed = False

for i in range(len(car.vision)):
    state.append(float(1))
state.append(0)
state.append(0)


current_checkpoint = 0
reward = 0

def run():
    global current_checkpoint, crashed, reward, event, state, last_checkpoint
    if RENDER:
        screen.fill((0, 0, 0))
    reward = -1/100

    if RENDER:
        for line in lines:
            pygame.draw.line(screen, (255, 255, 255), line[0], line[1])

        for line in car.vision:
            pygame.draw.line(screen, (0, 255, 0), line[0], line[1])

        for line in checkpoints:
            if line == checkpoints[current_checkpoint % len(checkpoints)]:
                pygame.draw.line(screen, (255, 0, 0), line[0], line[1])
            elif line == checkpoints[last_checkpoint % len(checkpoints)]:
                pygame.draw.line(screen, (125, 125, 0), line[0], line[1])

            else:
                pygame.draw.line(screen, (0, 0, 255), line[0], line[1])




    dy = -1 * math.degrees(car.rotation)
    car.imgcpy = pygame.transform.rotate(car.img, dy)
    car.center2 = (
        car.x - int(car.imgcpy.get_width() / 2), car.y - int(car.imgcpy.get_height() / 2))
    if RENDER:
        screen.blit(car.imgcpy, car.center2)

    if event == 1:
        if car.speed <= .5:
            car.speed += .5 / 8
    elif event == 2:
        car.rotation -= .005 * 2
    elif event == 3:
        car.rotation += .005 * 2
    elif event == 4:
        if car.speed > 0:
            car.speed -= .25/4
        elif car.speed >= -.25:
            car.speed -= .25/8
    elif event == 5:
        if car.speed <= .5:
            car.speed += .5 / 8
        car.rotation -= .005 * 2
    elif event == 6:
        if car.speed <= .5:
            car.speed += .5 / 8
        car.rotation += .005 * 2
    elif event == 7:
        if car.speed > 0:
            car.speed -= .25/4
        elif car.speed >= -.25:
            car.speed -= .25/8
        car.rotation -= .005 * 2
    elif event == 8:
        if car.speed > 0:
            car.speed -= .25/4
        elif car.speed >= -.25:
            car.speed -= .25/8
        car.rotation += .005 * 2
    else:
        if car.speed > 0:
            car.speed -= .25/8
        if car.speed < 0:
            car.speed += .25/8

    car.y -= car.speed * math.cos(car.rotation)
    car.x += car.speed * math.sin(car.rotation)

    car.update()

    for i in range(len(car.vision)):
        lowest = 1000
        for line2 in lines:
            point = line_intersection(car.vision[i], line2)
            distance = (car.center[0] - point[0])**2
            distance += (car.center[1] - point[1])**2
            distance = math.sqrt(distance)
            if distance < lowest:
                lowest = distance
            if distance < 13:
                crashed = True
                current_checkpoint = 0
                last_checkpoint = -2

        state[i] = lowest / 1000

    for i in range(len(car.vision)):
        point = line_intersection(car.vision[i], checkpoints[current_checkpoint % len(checkpoints)])
        distance = math.sqrt((car.center[0] - point[0])**2 + (car.center[1] - point[1])**2)
        if distance < 13:
            reward = 1
            current_checkpoint += 1
            last_checkpoint = current_checkpoint - 2

    for i in range(len(car.vision)):
        point = line_intersection(car.vision[i], checkpoints[last_checkpoint % len(checkpoints)])
        distance = math.sqrt((car.center[0] - point[0])**2 + (car.center[1] - point[1])**2)
        if distance < 13:
            reward = -5/100
            last_checkpoint -= 1
    center_of_next_cp_y = (checkpoints[current_checkpoint % len(checkpoints)][0][1] + checkpoints[current_checkpoint % len(checkpoints)][1][1])/2
    center_of_next_cp_x = (checkpoints[current_checkpoint % len(checkpoints)][0][0] + checkpoints[current_checkpoint % len(checkpoints)][1][0])/2
    rot_radian = math.atan2((center_of_next_cp_y - car.y), (center_of_next_cp_x - car.x))
    dy1 = -1 * math.degrees(rot_radian)

    rot_radian_car = math.atan2((car.vision[0][1][1] - car.y), (car.vision[0][1][0] - car.x))
    dy2 = -1 * math.degrees(rot_radian_car)

    dy = (dy2 - dy1) / 360
    state[-2] = dy
    state[-1] = car.speed

    if RENDER:
        pygame.display.update()

