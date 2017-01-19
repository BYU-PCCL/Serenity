#!/usr/bin/env python


import sys, random, math, pygame
from pygame.locals import *
from math import sqrt,cos,sin,atan2
from RRT_includes import *

#constants

EPSILON = 7.0
NUMNODES = 5000
GOAL_RADIUS = 10
MIN_DISTANCE_TO_ADD = 1.0
GAME_LEVEL = 1

white = 255, 240, 200
black = 20, 20, 40
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255
cyan = 0,255,255

class RRT:

    def __init__(self, screen, clock, xdim, ydim, paint=False):
        self.xdim = xdim
        self.ydim = ydim
        self.screen = screen
        self.clock = clock
        self.paint = paint
        self.count = 0
        self.rectObs = None

    def dist(self, p1,p2):
        return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

    def step_from_to(self, p1,p2):
        if dist(p1,p2) < EPSILON:
            return p2
        else:
            theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
            return p1[0] + EPSILON*cos(theta), p1[1] + EPSILON*sin(theta)

    def collides(self, pos):
        for points in self.rectObs:
            """
            This code is patterned after [Franklin, 2000]
            http://www.geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
            Tells us if the point is in this polygon
            """
            cn = 0  # the crossing number counter
            pts = points[:]
            pts.append(points[0])
            for i in range(len(pts) - 1):
                if (((pts[i][1] <= pos[1]) and (pts[i+1][1] > pos[1])) or ((pts[i][1] > pos[1]) and (pts[i+1][1] <= pos[1]))):
                        if (pos[0] < pts[i][0] + float(pos[1] - pts[i][1]) / (pts[i+1][1] - pts[i][1]) * (pts[i+1][0] - pts[i][0])):
                                cn += 1
            if bool(cn % 2)==1:
                return True
        return False

    def get_random(self):
        return random.random()*self.xdim, random.random()*self.ydim

    def get_random_clear(self):
        while True:
            p = self.get_random()
            noCollision = self.collides(p)
            if noCollision == False:
                return p

    def reset(self, obstacles):
        self.count = 0
        self.rectObs = obstacles

        if not self.screen == None:
            self.screen.fill(black)
            for p in obstacles:
                #pygame.draw.rect(self.screen, red, p)
                pygame.draw.polygon(self.screen, (255,255,255), p)


    def run(self, start, end, obstacles):

        path = []

        self.reset(obstacles)

        initPoseSet = False
        initialPoint = Node(None, None)
        goalPoseSet = False
        goalPoint = Node(None, None)
        currentState = 'init'
        nodes = []

        #SET START POINT
        initialPoint = Node(start, None)
        nodes.append(initialPoint) # Start in the center
        initPoseSet = True
        if self.paint:
            pygame.draw.circle(self.screen, green, initialPoint.point, GOAL_RADIUS)


        #SET END POINT
        goalPoint = Node(end,None)
        goalPoseSet = True
        if self.paint:
            pygame.draw.circle(self.screen, blue, goalPoint.point, GOAL_RADIUS)
        currentState = 'buildTree'

        path.append(end)
        pathNotFound = True
        #reset()
        while pathNotFound:
            if currentState == 'goalFound':
                #traceback
                
                currNode = goalNode.parent
                while currNode.parent != None:
                    if self.paint:
                        pygame.draw.line(self.screen,cyan,currNode.point,currNode.parent.point)
                    path.append(currNode.point)
                    #print currNode.point
                    currNode = currNode.parent
                path.append(initialPoint.point)
                optimizePhase = True
                pathNotFound = False

            elif currentState == 'buildTree':
                self.count = self.count+1
                if self.count < NUMNODES:
                    foundNext = False
                    while foundNext == False:
                        rand = self.get_random_clear()
                        # print("random num = " + str(rand))
                        parentNode = nodes[0]

                        for p in nodes: #find nearest vertex
                            if self.dist(p.point,rand) <= self.dist(parentNode.point,rand): #check to see if this vertex is closer than the previously selected closest
                                newPoint = self.step_from_to(p.point,rand)
                                if self.collides(newPoint) == False: # check if a collision would occur with the newly selected vertex
                                    parentNode = p #the new point is not in collision, so update this new vertex as the best
                                    foundNext = True

                    newnode = self.step_from_to(parentNode.point,rand)
                    nodes.append(Node(newnode, parentNode))
                    if self.paint:
                        pygame.draw.line(self.screen,white,parentNode.point,newnode)

                    if point_circle_collision(newnode, goalPoint.point, GOAL_RADIUS):
                        currentState = 'goalFound'
                        goalNode = nodes[len(nodes)-1]

                    if self.count%100 == 0 and self.paint:
                        print("node: " + str(self.count))
                else:
                    print("Ran out of nodes... :(")
                    return;

            #handle events
            if self.paint:
                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Exiting")
                
            if self.paint:
                pygame.display.update()
                self.clock.tick(10000)
        
        
        return path



