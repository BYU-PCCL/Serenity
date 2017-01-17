import math

#
# In world.py :
# 	self.isovist = isovist.Isovist(self.terrain)
#
# In find.py :
# 	drone_isovist = w.isovist.FindIsovistForAgent(c.x, c.y)
#    
#   # to show the corners in white
#   for point in drone_isovist:
#		#print point
#		x = point[0]
#		y = point[1]
#		pygame.draw.rect(screen, GREEN, [x - ICON_SIZE/4, y - ICON_SIZE/4, ICON_SIZE/2, ICON_SIZE/2])
#
#
#

class Isovist:

	def __init__(self, array, arow=-1, acol=-1):
		self.array = array
		self.corners = []
		self.arow = arow
		self.acol = acol
		#self.PrintArray()
		self.FindCorners()
		self.AddBorderToCorners()
		#self.MarkCorners()

	def PrintArray(self):
		print "[Printing Array]"
		for x in xrange(len(self.array)):
			print self.array[x]

	def AddBorderToCorners(self):
		#adding first and last row
		for i in xrange(len(self.array[0])):
			self.corners.append((0, i))
			self.corners.append((len(self.array)-1, i))
		#adding first and last col
		for i in xrange(len(self.array)):
			self.corners.append((i, 0))
			self.corners.append((i, len(self.array[0])-1))

	def FindCorners(self):
		for row in xrange(1,len(self.array)):
			for col in xrange(1,len(self.array[row])):
				#print col, row
				focus = []
				focus.append([ self.array[row-1][col-1], self.array[row-1][col]])
				focus.append([ self.array[row][col-1], self.array[row][col]])

				oneCounts = 0
				for f in focus:
				#	print f
					oneCounts += f.count(1)

				if oneCounts == 1 or oneCounts == 3:
					#print [(ix,iy) for ix, r in enumerate(focus) for iy, i in enumerate(r) if i == 1]

					if oneCounts == 3:
						locs = [(ix,iy) for ix, r in enumerate(focus) for iy, i in enumerate(r) if i == 0]
						#array[row -1 + locs[0][0]][ col -1 + locs[0][1]] = 2
						self.corners.append((row -1 + locs[0][0], col -1 + locs[0][1] ))

					if oneCounts == 1:
						locs = [(ix,iy) for ix, r in enumerate(focus) for iy, i in enumerate(r) if i == 1]
						
						#array[row -1 + locs[0][0]][ col -1 + locs[0][1]] = 2
						left = locs[0][0]
						right = locs[0][1]

						if right == 1:
							right = 0
						else:
							right = 1

						if left == 1:
							left = 0
						else:
							left = 1

						locs = [(left, right)]

						self.corners.append((row -1 + locs[0][0], col -1 + locs[0][1] ))
	
	
	def MarkCorners(self):
		print "\nMarking Corners:"
		for corner in self.corners:
			#print corner
			array[corner[0]][corner[1]] = 4

		#self.PrintArray()

	def FindIsovistForAgent(self, arow, acol):
		self.arow = arow
		self.acol = acol
		#show agent 
		#if showAgent:
		#	array[arow][acol] = 5

		isovistPoints = []
		for c in self.corners:
			#print "corner:" , c
			#get corners
			crow = c[0]
			ccol = c[1]

			#find vector 
			vrow = crow - arow
			vcol = ccol - acol

			#find distance/magnitude
			#print "vector:", vrow, vcol

			magnitude = ((vrow ** 2) + (vcol ** 2)) ** 0.5
			#print "mag:",magnitude
			if magnitude != 0:
				#normalizing the vector to 1 step
				normrow = vrow / magnitude
				normcol = vcol / magnitude

				#print "norm:" , normrow, normcol
				#normrow *= .7
				#normcol *= .7
				#print "norm:", normrow, normcol

				clear = True
				curr_step_row = arow
				curr_step_col = acol

				step_loc_row = int(round(curr_step_row))
				step_loc_col = int(round(curr_step_col))

				while not (step_loc_row, step_loc_col) == (crow, ccol):
					#take a step
					curr_step_row += normrow
					curr_step_col += normcol

					step_loc_row = int(round(curr_step_row))
					step_loc_col = int(round(curr_step_col))

					#print "step:", step_loc_row, step_loc_col
					#array[step_loc_row][step_loc_col] = 8
					if self.array[step_loc_row][step_loc_col] == 1:

						#exclude walls as obsacles 
						if step_loc_row not in [0,len(self.array)-1] and step_loc_col not in [0,len(self.array[0])-1]:
							clear = False
						break
					#
				if clear:
					#array[crow][ccol] = 7
					isovistPoints.append((crow,ccol))

		#sort in circular manner
		#self.PrintArray()
		#print "Points:", isovistPoints
		return self.SortIntoPolygonPoints(isovistPoints)

	def SortIntoPolygonPoints(self, points):
		#print "Points:", points
		points.sort(self.compare)
		return points


	def compare(self, a, b):
		a_row = a[0]
		a_col = a[1]

		b_row = b[0]
		b_col = b[1]

		a_vrow = a_row - self.arow
		a_vcol = a_col - self.acol

		b_vrow = b_row - self.arow
		b_vcol = b_col - self.acol

		a_ang = math.degrees(math.atan2(a_vrow, a_vcol))
		b_ang = math.degrees(math.atan2(b_vrow, b_vcol))

		#print "angle:", a_ang, b_ang

		#print math.degrees(math.atan2(0)), math.degrees(math.atan2(1)), math.degrees(math.atan2(-1))

		if a_ang < b_ang:
			return -1

		if a_ang > b_ang:
			return 1

		return 0 








# # Testing main

# array = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
# 		 [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
# 		 [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
# 		 [1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1],
# 		 [1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1],
# 		 [1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1],
# 		 [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# 		 [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# 		 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

# arow = 8
# acol = 11
# iso = Isovist(array)
# #iso.PrintArray()



# isovistPoints = iso.FindIsovistForAgent(arow, acol, 1)

# print isovistPoints

# iso.PrintArray()

# count = 2
# for p in isovistPoints:
# 	iso.array[p[0]][p[1]] = count
# 	count += 1

# iso.PrintArray()








