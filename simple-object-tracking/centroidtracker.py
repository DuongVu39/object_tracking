# import the necessary packages
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist
from collections import OrderedDict
from typing import List, Optional
import numpy as np


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        """initialize the next unique object ID along with two ordered
        dictionaries used to keep track of mapping a given object
        ID to its centroid and number of consecutive frames it has
        been marked as "disappeared", respectively"""

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to delete the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid: List):
        """Register new object with its centroid"""

        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0 # mark the number of frame the object is not there (right now is 0)
        self.nextObjectID += 1

    def deregister(self, objectID: int):
        """Deregister object if it disappeared from frame for more than max threshold
            To deregister an object ID we delete the object ID from both of our respective dictionaries"""

        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects: List[List]):
        """update object list if there are new object appeared
        Arg:
            rects: List of list, elements is bounding box rectangles (feed in by the object detector)
                   example: [(startX, startY, endX, endY), ...]
        """

        # check if the list of bounding box is empty
        if len(rects) == 0:
            # loop through the list of all existing object and mark them as disappeared
            # by adding 1 (number of frame it is disappearing)
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if the number of frame disappear exceed the max threshold, deregister the object:
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # If there's no object tracked before, return blank object list
            return self.objects

        # get list of centroid from the input bounding boxes:
        inputCentroids = list()

        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids.append((cX, cY))

        # if there's no object tracked before, simply register all new object with its centroid
        if len(self.objects) == 0:
            for obj in inputCentroids:
                self.register(obj)

        # else need to figure which one is already in the list to track.
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            dist_matrix = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Using the Hungarian algorithm (linear_sum_assignment) to calculate the minimum distance between centroids
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(row_ind, col_ind):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedCols = set(range(dist_matrix.shape[1])).difference(usedCols)
            unusedRows = set(range(dist_matrix.shape[0])).difference(usedRows)

            # Set case when there are more new objects than current tracked objects
            if dist_matrix.shape[0] < dist_matrix.shape[1]:
                for col in unusedCols:
                    self.register(inputCentroids[col])

            # In case there are more tracked objects than new objects:
            else:
                # Then increase the disappeared frame by 1
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                # Then check if the number of disappeared frame exceed the threshold:
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

        # return the set of tracked objects:
        return self.objects
