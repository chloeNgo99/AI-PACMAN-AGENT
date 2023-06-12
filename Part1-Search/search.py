# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfdirection(self, direction):
        """
         direction: A list of direction to take

        This method returns the total cost of a particular sequence of direction.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of direction that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("graph ", problem)
    stack = util.Stack()
    stack.push((problem.getStartState(), []))
    visited = []
    while not stack.isEmpty():
        value, direction = stack.pop()
        #if found the goal, return the direction
        if problem.isGoalState(value):
            return direction
        #if not visited, add to visited and push the successors
        if value not in visited:
            visited.append(value)
            for successor, action, cost in problem.getSuccessors(value):
                stack.push((successor, direction + [action]))
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    seen = []
    fringe.push((problem.getStartState(), [], 0))

    while not fringe.isEmpty():
        state, actions, cost = fringe.pop()
        if problem.isGoalState(state):
            return actions
        if state not in seen:
            seen.append(state)
            for child, action, cost in problem.getSuccessors(state):
                if child not in seen:
                    action = actions + [action]
                    fringe.push((child, action, cost))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((0, start, []), 0)
    visit = set()
    #print("graph ", problem)
    while not frontier.isEmpty():
        pathCost, value, direction = frontier.pop()
       
        if problem.isGoalState(value):
            return direction

        if value not in visit:
            visit.add(value)
            for successor, action, step_cost in problem.getSuccessors(value):
                newPathCost  = pathCost + step_cost
                newDirection = direction + [action]
                frontier.push((newPathCost, successor, newDirection), newPathCost)

    # Return failure if frontier is empty
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    seen = []
    fringe.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))
    while not fringe.isEmpty():
        state, actions, cost = fringe.pop()
        if problem.isGoalState(state):
            return actions
        if state not in seen:
            seen.append(state)
            for child, action, cost in problem.getSuccessors(state):
                if child not in seen:
                    action = actions + [action]
                    cost = problem.getCostOfActions(action)
                    fringe.push((child, action, cost), cost + heuristic(child, problem))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
