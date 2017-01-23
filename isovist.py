import math
import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys

'''

USAGE:
    import isovist as iso
    isovist = iso.Isovist(polygonSegments) # See **BELOW** for Data Structure of MAP
    
    isIntruderFound = isovist.IsIntruderSeen(RRTPath, UAVLocation, UAVForwardVector, UAVFieldOfVision = 45)
 

EXAMPLE:
    insovist_main.py 


**BELOW**:

    # Data Structure for MAP
    polygonSegments = []
    
    #One Polygon 
    polygonSegments.append([ 
        [ (0,0),(840,0) ], 
        [ (840,0),(840,360) ],
        [ (840,360), (0,360)],
        [ (0,360), (0,0) ]
        ])   


'''

class Isovist:

    def __init__(self, polygon_map):
        self.polygon_map = polygon_map
        self.uniquePoints = self.GetUniquePoints()


    def IsIntruderSeen(self, RRTPath, UAVLocation, UAVForwardVector, UAVFieldOfVision = 40):
        self.fieldOfVision = math.radians(UAVFieldOfVision/2.0)
        self.forwardVector = UAVForwardVector

        intersections = self.GetIsovistIntersections(UAVLocation, UAVForwardVector)
        #print "\n\n ISOVIST:", intersections

        for point in RRTPath:
            isFound = self.FindIntruderAtPoint(point, intersections)
            if isFound:
                return True
        return False


    def FindIntruderAtPoint(self, pos, intersections):
        points = intersections
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

    def GetIsovistIntersections(self, agentLocation, direction):
        self.agentLocation = agentLocation
        uniqueAngles = self.GetUniqueAngles(direction)

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
        intersections.insert(0, agentLocation)
        return intersections

    def GetUniqueAngles(self, direction):
        alpha, beta = self.GetAgentViewRays(direction)
        alphaAngle = math.atan2(alpha[1]-self.agentLocation[1], alpha[0]-self.agentLocation[0])
        betaAngle = math.atan2(beta[1]-self.agentLocation[1], beta[0]-self.agentLocation[0])
        self.startingFieldOfVision = alpha
        uniqueAngles = [alphaAngle, betaAngle]
        for point in self.uniquePoints:
            angleBetween = self.GetRelativeAngle(point, direction)

            if math.fabs(angleBetween) <= self.fieldOfVision:
                #find world angle
                angle = math.atan2(point[1]-self.agentLocation[1], point[0]-self.agentLocation[0])
                uniqueAngles.append(angle)
                uniqueAngles.append(angle-0.01)
                uniqueAngles.append(angle+0.01)

        return uniqueAngles

    def GetRelativeAngle(self, point, direction):
        #forward direction
        dx = direction[0] 
        dy = direction[1]
        dMag = math.sqrt(dx**2 + dy**2)

        #vector agentlocation to point
        uniqueVectorX = point[0] - self.agentLocation[0]
        uniqueVectorY = point[1] - self.agentLocation[1]
        uniqueMag = math.sqrt(uniqueVectorX**2 + uniqueVectorY**2)

        #dot product equation stuff to find angle between the two vectors
        dotProduct = uniqueVectorX * dx + uniqueVectorY * dy
        cosineTheta = dotProduct / (uniqueMag * dMag)
        cosineTheta = min(1,max(cosineTheta,-1))

        angleBetween = math.acos(cosineTheta)
        return angleBetween
        
    def SortIntoPolygonPoints(self, points):
        points.sort(self.Compare)
        return points

    def Compare(self, a, b):
        a_ang = self.GetRelativeAngle(a, self.startingFieldOfVision)
        b_ang = self.GetRelativeAngle(b, self.startingFieldOfVision)

        if a_ang < b_ang:
            return -1
        if a_ang > b_ang:
            return 1
        return 0 

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


    def GetAgentViewRays(self, direction):
        dx = direction[0] * 2000
        dy = direction[1] * 2000

        alphax = dx * math.cos(self.fieldOfVision) - dy * math.sin(self.fieldOfVision)
        alphay = dy * math.cos(self.fieldOfVision) + dx * math.sin(self.fieldOfVision)

        betax = dx * math.cos(-self.fieldOfVision) - dy * math.sin(-self.fieldOfVision)
        betay = dy * math.cos(-self.fieldOfVision) + dx * math.sin(-self.fieldOfVision)
        return (alphax, alphay), (betax, betay)




    







