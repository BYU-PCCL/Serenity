import math
import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys



class Isovist:

    def __init__(self, polygon_map, agentLocation=None):

        self.agentLocation = agentLocation
        self.polygon_map = polygon_map
        self.uniquePoints = self.GetUniquePoints()

    def GetIntersection(self, ray, segment):
        
        # RAY in parametric: Point + Direction * T1
        r_px = ray[0][0]
        r_py = ray[0][1]

        # direction
        r_dx = ray[1][0] - ray[0][0]
        r_dy = ray[1][1] - ray[0][1]

        # SEGMENT in parametric: Point + Direction*T2
        s_px = segment[0][0]
        s_py = segment[0][1]

        # direction
        s_dx = segment[1][0] - segment[0][0]
        s_dy = segment[1][1] - segment[0][1]

        
        r_mag = math.sqrt(r_dx ** 2 + r_dy ** 2)
        s_mag = math.sqrt(s_dx ** 2 + s_dy ** 2)



        # PARALLEL - no intersection
        if (r_dx/r_mag) == (s_dx/s_mag):
            if (r_dy/r_mag) == (s_dy/s_mag):
                return None, None
        

        denominator = float( -s_dx*r_dy + r_dx*s_dy )
        if denominator == 0:
            return None, None

        T1 = (-r_dy * (r_px - s_px) + r_dx * ( r_py - s_py)) / denominator
        T2 = (s_dx * ( r_py - s_py) - s_dy * ( r_px - s_px)) / denominator


        if T1 >= 0 and T1 <= 1 and T2 >= 0 and T2 <= 1:
            #Return the POINT OF INTERSECTION
            x = r_px+r_dx*T2
            y = r_py+r_dy*T2
            param = T2
            return (int(x),int(y)), param

        return None, None


    def GetUniquePoints(self):
        points = []
        for polygon in self.polygon_map:
            for segment in polygon:
                if segment[0] not in points:
                    points.append(segment[0])
                if segment[1] not in points:
                    points.append(segment[1])
        return points

    def GetUniqueAngles(self):
        uniqueAngles = []
        for point in self.uniquePoints:
            angle = math.atan2(point[1]-self.agentLocation[1], point[0]-self.agentLocation[0])
            uniqueAngles.append(angle)
            uniqueAngles.append(angle-0.0001)
            uniqueAngles.append(angle+0.0001)
        return uniqueAngles

    def SortIntoPolygonPoints(self, points):
        points.sort(self.compare)
        return points


    def compare(self, a, b):

        a_row = a[0]
        a_col = a[1]

        b_row = b[0]
        b_col = b[1]

        a_vrow = a_row - self.agentLocation[0]
        a_vcol = a_col - self.agentLocation[1]

        b_vrow = b_row - self.agentLocation[0]
        b_vcol = b_col - self.agentLocation[1]

        a_ang = math.degrees(math.atan2(a_vrow, a_vcol))
        b_ang = math.degrees(math.atan2(b_vrow, b_vcol))

        if a_ang < b_ang:
            return -1

        if a_ang > b_ang:
            return 1

        return 0 

    def GetIntersections(self, agentLocation):
        self.agentLocation = agentLocation

        uniqueAngles = self.GetUniqueAngles()

        intersections = []
        for angle in uniqueAngles:

            # Calculate dx & dy from angle
            dx = math.cos(angle) * 2000
            dy = math.sin(angle) * 2000

            # Ray from center of screen to mouse
            ray = [ agentLocation , (agentLocation[0]+dx, agentLocation[1]+dy) ]

            # Find CLOSEST intersection
            closestIntersect = None
            closestParam = 10000000

            for polygon in self.polygon_map:
                for segment in polygon:
                    intersect, param = self.GetIntersection(ray, segment)
                    
                    if intersect != None:
                        if closestIntersect == None or param < closestParam:
                            closestIntersect = intersect
                            closestParam = param
            
            if closestIntersect != None:
                intersections.append(closestIntersect)

        intersections = self.SortIntoPolygonPoints(intersections)
        return intersections
