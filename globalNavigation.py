""" Code taken from the pyvisgraph as is along with explanations, with additional explanations where needed
https://github.com/TaipanRex/pyvisgraph.git
"""

"""
The MIT License (MIT)

Copyright (c) 2016 Christian August Reksten-Monsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict
import numpy as np
import math
import heapq

class Point(object):
    __slots__ = ('x', 'y', 'polygon_id')

    def __init__(self, x, y, polygon_id=-1):
        self.x = float(x)
        self.y = float(y)
        self.polygon_id = polygon_id

    def __eq__(self, point):
        return point and self.x == point.x and self.y == point.y

    def __ne__(self, point):
        return not self.__eq__(point)

    def __lt__(self, point):
        """ This is only needed for shortest path calculations where heapq is
            used. When there are two points of equal distance, heapq will
            instead evaluate the Points, which doesnt work in Python 3 and
            throw a TypeError."""
        return hash(self) < hash(point)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def __hash__(self):
        return self.x.__hash__() ^ self.y.__hash__()

    def __repr__(self):
        return "Point(%.2f, %.2f)" % (self.x, self.y)

class Edge(object):
    __slots__ = ('p1', 'p2')

    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2

    def get_adjacent(self, point):
        if point == self.p1:
            return self.p2
        return self.p1

    def __contains__(self, point):
        return self.p1 == point or self.p2 == point

    def __eq__(self, edge):
        if self.p1 == edge.p1 and self.p2 == edge.p2:
            return True
        if self.p1 == edge.p2 and self.p2 == edge.p1:
            return True
        return False

    def __ne__(self, edge):
        return not self.__eq__(edge)

    def __str__(self):
        return "({}, {})".format(self.p1, self.p2)

    def __repr__(self):
        return "Edge({!r}, {!r})".format(self.p1, self.p2)

    def __hash__(self):
        return self.p1.__hash__() ^ self.p2.__hash__()

class Graph(object):
    """
    A Graph is represented by a dict where the keys are Points in the Graph
    and the dict values are sets containing Edges incident on each Point.
    A separate set *edges* contains all Edges in the graph.

    The input must be a list of polygons, where each polygon is a list of
    in-order (clockwise or counter clockwise) Points. If only one polygon,
    it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
    Point(2,1)]].

    *polygons* dictionary: key is a integer polygon ID and values are the
    edges that make up the polygon. Note only polygons with 3 or more Points
    will be classified as a polygon. Non-polygons like just one Point will be
    given a polygon ID of -1 and not maintained in the dict.
    """

    def __init__(self, polygons):
        self.graph = defaultdict(set)
        self.edges = set()
        self.polygons = defaultdict(set)
        pid = 0
        for polygon in polygons:
            if polygon[0] == polygon[-1] and len(polygon) > 1:
                polygon.pop()
            for i, point in enumerate(polygon):
                sibling_point = polygon[(i + 1) % len(polygon)]
                edge = Edge(point, sibling_point)
                if len(polygon) > 2:
                    point.polygon_id = pid
                    sibling_point.polygon_id = pid
                    self.polygons[pid].add(edge)
                self.add_edge(edge)
            if len(polygon) > 2:
                pid += 1

    def get_adjacent_points(self, point):
        return [edge.get_adjacent(point) for edge in self[point]]

    def get_points(self):
        return list(self.graph)

    def get_edges(self):
        return self.edges
    
    def get_polygons(self):
        return self.polygons

    def add_edge(self, edge):
        self.graph[edge.p1].add(edge)
        self.graph[edge.p2].add(edge)
        self.edges.add(edge)

    def __contains__(self, item):
        if isinstance(item, Point):
            return item in self.graph
        if isinstance(item, Edge):
            return item in self.edges
        return False

    def __getitem__(self, point):
        if point in self.graph:
            return self.graph[point]
        return set()

    def __str__(self):
        res = ""
        for point in self.graph:
            res += "\n" + str(point) + ": "
            for edge in self.graph[point]:
                res += str(edge)
        return res

    def __repr__(self):
        return self.__str__()


# 0 - collinear
# 1 - clockwise
# -1 - counterclockwise

def orientation(p1, p2, p3):

    """ Determines the orientation of the three ordered points

    Args:
        p1 (Point): The first point 
        p2 (Point): The second point
        p3 (Point): The third point

    Return:
        int: represents the orientation based on the following convention
           0 - collinear
           1 - clockwise
           -1 - counterclockwise

    """

    angle = (float)(p2.y - p1.y)*(p3.x-p2.x) - (float)(p3.y-p2.y)*(p2.x-p1.x) # difference of line segment slopes, only the sign is important 
    if(angle < 0):
        return -1
    if angle > 0:
        return 1
    return 0

def on_segment(p, q, r): 

    """ Determines whether the point is on the segment 

    Args:
        p (Point): The first point of the segment
        q (Point): The point inquiry is about
        r (Point): The second point of the segment

    Return:
        bool: point is on the segment

    """

    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False

def intersect(edge, obstacle):

    """ Determines whether two line segments are intersecting

    Args:
        edge (Edge): The first line segment 
        obstacle (Edge): The second line segment

    Return:
        bool: two line segments intersect

    """

    if(obstacle.p1 in edge) or (obstacle.p2 in edge): # common point is not considered an intersection in this sense
        return False
    
    o1 = orientation(edge.p1, edge.p2, obstacle.p1)
    o2 = orientation(edge.p1, edge.p2, obstacle.p2)
    o3 = orientation(obstacle.p1, obstacle.p2, edge.p1)
    o4 = orientation(obstacle.p1, obstacle.p2, edge.p2)
    #general case 
    if (o1 != o2) and (o3 != o4):
        return True
    if (o1 == 0) and on_segment(edge.p1, obstacle.p1, edge.p2):
        return True
    if (o2 == 0) and on_segment(edge.p1, obstacle.p2, edge.p2):
        return True
    if (o3 == 0) and on_segment(obstacle.p1, edge.p1, obstacle.p2):
        return True
    if (o4 == 0) and on_segment(obstacle.p1, edge.p2, obstacle.p2):
        return True
    
    return False

"""
    Code below is inspired by the pyvisgraph library
"""

def point_in_polygon(p1, polygon):
    
    """ Checks if a point is inside a polygon
    
        Checks if the number of intersections of a half-line from p1 to (-inf, p1.y) and the polygon,
        if it's an odd number of intersections the point is inside the polygon

    Args:
        p1 (Point): The point 
        polygon ([Edge]): The polygon

    Returns:
        bool: Point is in polygon
        
    """
    p2 = Point(100000, p1.y)
    line = Edge(p1, p2)
    count = 0   
    for edge in polygon:
        if edge.p1 == p1 or edge.p2 == p1:
            return False
        
        if intersect(line, edge):
            if orientation(edge.p1, edge.p2, p1) == 0:
                return on_segment(edge.p1, p1, edge.p2)  
            count += 1
            
    if count % 2 == 1:
        return True
    return False

def edge_in_polygon(p1, p2, graph):

    """ Determines whether an edge is inside a polygon
        
    Args:
        p1 (Point): The first point of the edge
        p2 (Point): The second point of the edge
        graph (Graph): The graph we are searching
        
    Return:
        bool: edge is in the polygon

    """
    
    if p1.polygon_id != p2.polygon_id: # points are not part of the same polygon
        return False
    if p1.polygon_id == -1 or p2.polygon_id == -1: # it's the starting point or the goal point 
        return False

    mid_point = Point((p1.x + p2.x)/2, (p1.y+p2.y)/2)
    return point_in_polygon(mid_point, graph.get_polygons()[p1.polygon_id])


def visible_vertices(point, graph, start = None, goal = None):
    
    """ Finds all the vertices a point sees
    
    Args:
        point (Point): the point whose visibile vertices we are looking for
        graph (Graph): the graph that captures the free space of the world map
        start (Point): optional parameter
                        in case there is a navigation problem defined from the start, if not vertices visible from the start
                        and the goal will be added afterwards, additionally it is known that if a point p1 sees point p2, point p2 sees point p1
        goal (Point):  optional parameter
                        similar as for the start

    Returns:
        list: of all vertices, points, in the graph, or the union of the graph and start and/or goal point, visible from the point point
        
    """
    
    obstacles = [] # consists of all edges of the polygons - obstacles
    for polygons in graph.get_polygons().values():
        obstacles += polygons
        
    points = graph.get_points()
    if start:
        points.append(start)
    if goal:
        points.append(goal)
        
    visible = []
    for polygon in graph.get_polygons().values():
        if point_in_polygon(point, polygon):
            return []
                
    for other_point in points: # looping over the other n-1 vertcies, assuming that n is the total number of vertices
        in_poligon = False
        for polygon in graph.get_polygons().values():
            if point_in_polygon(other_point, polygon):
                in_poligon = True
                break
        if in_poligon:
            continue
                
        if point == other_point: 
            continue
        edge = Edge(point, other_point) 
        if edge in graph:
            """
                if it is already part of the graph, it's either an edge of the obstacle, since it's enlarged we can traverse it, 
                and if it's not it was obtained when checking the edge when the roles of points were reversed    
            """
            continue
        intersects = False
        
        for obstacle in obstacles: # looping over all of the obstacles
            if(intersect(edge, obstacle)): 
                """
                    checking the intersection between the edges of polygons and the edge of the point and 
                    the other point we are looking at right now
                """
                intersects = True
                break
        if intersects != False:
            continue
        # check if the visible edge is interior to its polygon
        if other_point not in graph.get_adjacent_points(point):
            """
                checking if the visible edge is interior to its polygon
            """
            if edge_in_polygon(point, other_point, graph):
                continue
            
        visible.append(other_point)
    return visible

def build_graph(graph, start = None, goal = None):
    """ Builds the graph based on the polygons from the global map

        Naive implementation - it loops over all of the vertices and checks whether any of 
        the remaining edges are visible
        
        - note that because of how function add_edges in class Graph works it will actually add edge from point p1 to point p2, 
        and point p2 to point p1 at the same time, when it is checking p1
        
        Args:
            graph (Graph): the graph which will be built, at this point only consisting of the polygons and their edges,
                            also the variable in which the result will be stored in
            start (Point): optional parameter 
                            in case there is a navigation problem defined from the start, if not vertices visible from the start and 
                            the goal will be added afterwards, additionally it is known that if a point p1 sees point p2, point p2 sees point p1
            goal (Point):  optional parameter
                            similar to the start 
    
    """
    
    points = graph.get_points()
    if start:
        points.append(start)
    if goal:
        points.append(goal)
        
    for point in points:
        """
            finding all the vertices visible from point and adding the visible edges to the graph
        """
        visible = visible_vertices(point, graph, start, goal) 
        for p in visible:
            graph.add_edge(Edge(point, p))
            
    return graph

"""
    Astar algorithm implementation
"""
def distanceBetweenPoints(p1, p2):
    """ Calculates the Eucledian distance between points

    Args:
        p1 (Point): The first point
        p2 (Point): The second point

    Returns:
        float: eucledian distance between points p1 and p2
    """
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def heuristic(point, goal):
    """ Heuristic function
    
    Heuristic function, distance from 

    Args:
        point (Point): The point
        goal (Point): The goal

    Returns:
        double: distance between the point and the goal
        
    """
    return distanceBetweenPoints(point, goal)

def Astar(graph, start, goal, frame_size):
    """ A* algorithm

    Args:
        graph (Graph): Graph that captures the conectivity of the free space in the global map
        start (Point): The starting point
        goal (Point): The goal point
        frame_size (2D array): Size of the frame

    Returns:
        list, list: list of nodes of the plan in reverse order, list of all explored nodes
    """
    
    # checking if start and goal are valid points 
    for polygon in graph.get_polygons().values():
        if point_in_polygon(start, polygon) or point_in_polygon(goal, polygon):
            return [], []
    
    start_exists = start in graph
    goal_exists = goal in graph
    add_to_graph = Graph([])
    
    # polygon for the playground
    p1 = Point(0,0)
    p2 = Point(frame_size[0],0)
    p3 = Point(frame_size[0],frame_size[1])
    p4 = Point(0, frame_size[1])
    frame = Graph([[p1, p2, p3, p4]])
    
    # checking conectivity of the start and goal with the vertices of the graph
    if not start_exists:
        for v in visible_vertices(start, graph, goal=goal):
            add_to_graph.add_edge(Edge(start, v))
    if not goal_exists:
        for v in visible_vertices(goal, graph):
            add_to_graph.add_edge(Edge(goal, v))

    points = graph.get_points()
    opened = [(heuristic(start,goal),start)] # storing a node as cost of path and the node itself, to accomodate for the use of heapify
    heapq.heapify(opened) # storing all of the opened nodes in a heap, to access the best node faster and easier
    closed = [] # all of the nodes that have been explored so far
    cameFrom = dict() # key = node, value = parent node
    costs = dict(zip(points, [np.inf for x in range(len(points))])) # key = node, value = cost from the start to node
    costs[start] = 0

    nodes = [] # plan initialized
    
    print("A*")
    while len(opened) != 0:
        currentEstimate, current = heapq.heappop(opened) # get the best node, smallest cost value
        closed.append(current) 
        if current == goal: 
            """
                if the current node is the goal the path has been found and it needs to be extracted using the cameFrom,
                which captures the parent-child relationships that the algorithm creates
            """
            while current != start:
                nodes.append(current)
                temp = cameFrom[current]
                current = temp
            nodes.append(current)
            
            return nodes, closed
        
        for neighbour in (graph.get_adjacent_points(current) + add_to_graph.get_adjacent_points(current)): # checks all the neighbouring points
            """if not point_in_polygon(neighbour, frame.get_polygons()[0]): # if point not in frame don't take it into consideration
                print("current", current)
                print("neighbour ", neighbour)
                print(point_in_polygon(neighbour, frame.get_polygons()[0]))
                print("skipped")
                #return [],[]
                continue"""
            cost = costs[current] + heuristic(current, neighbour)
            if neighbour in costs: 
                """
                    if there was a path to the neighbouring point, no matter if was or wasn't visited, 
                    if the cost to go through the current node is smaller than the existing the neighbour should be visited again
                """
                if cost >= costs[neighbour]:
                    continue
            # adding the node and updating the parent-child dictionary and costs
            opened.append((cost + heuristic(neighbour, goal), neighbour))
            cameFrom[neighbour] = current
            costs[neighbour] = cost           
    print("No path found to goal")
    return nodes, closed

def global_navigation(polys, start, goal, frame_dimensions):
    """ Global navigation function
    
        Creates the global map and finds the optimal path

    Args:
        polys ([[Points]]]): The obstacles
        start (Point): The start point
        goal (Point): The goal point
        frame_dimensions (2D array): dimensions of the frame

    Returns:
        list: of positions
    """

    graph = Graph(polys)
    build_graph(graph)
    
    nodes, closed = Astar(graph, start, goal, frame_dimensions)
    
    #for node in nodes:
    nodes = nodes[::-1]
    coordinates = np.array(nodes)
    
    points = np.zeros((len(coordinates),2))
    
    for i in range(len(coordinates)):
        points[i]= [coordinates[i].x,coordinates[i].y]
        
    return points