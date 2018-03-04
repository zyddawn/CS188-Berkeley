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

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
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

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    start_state = problem.getStartState()
    #print "Start:", start_state
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    #print "\n"
    if problem.isGoalState(start_state):
        return []
    fringe = util.Stack()    # FILO
    closed_set = set()
    fringe.push((start_state, []))

    while not fringe.isEmpty():
        cur_state, solu = fringe.pop()
        #print "Current: ", cur_state
        #print "Is current a goal?", problem.isGoalState(cur_state)
        #print "Current's successors:", problem.getSuccessors(cur_state)
        #print "\n"
        if not cur_state in closed_set:
            closed_set.add(cur_state)
        else:
            continue
        if problem.isGoalState(cur_state):
            return solu

        succs = problem.getSuccessors(cur_state)
        for succ_state in succs:
            if not succ_state in closed_set:
                aug_solu = solu + [succ_state[1]]
                fringe.push((succ_state[0], aug_solu))



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    fringe = util.Queue()    # FIFO
    closed_set = set()
    fringe.push((start_state, []))
    #print start_state

    while not fringe.isEmpty():
        cur_state, solu = fringe.pop()
        #print cur_state
        #print solu

        if not cur_state in closed_set:
            closed_set.add(cur_state)
        else:
            continue
        if problem.isGoalState(cur_state):
            return solu

        succs = problem.getSuccessors(cur_state)
        for succ_state in succs:
            if not succ_state in closed_set:
                aug_solu = solu + [succ_state[1]]
                fringe.push((succ_state[0], aug_solu))



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    fringe = util.PriorityQueue()   # pop least accumulative cost node
    closed_set = set()
    fringe.push((start_state, []), 0)

    while not fringe.isEmpty():
        cur_state, cur_solu = fringe.pop()
        
        if not cur_state in closed_set:
            closed_set.add(cur_state)
        else:
            continue
        if problem.isGoalState(cur_state):
            return cur_solu

        succs = problem.getSuccessors(cur_state)
        for succ_state in succs:
            if succ_state not in closed_set:
                upd_solu = cur_solu + [succ_state[1]]
                fringe.update((succ_state[0], upd_solu), problem.getCostOfActions(upd_solu)) # update fringes 
        


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    fringe = util.PriorityQueue()   # pop least accumulative cost node
    closed_set = set()
    start_cost = 0 + heuristic(start_state, problem)   # A* cost
    fringe.push((start_state, []), start_cost)

    while not fringe.isEmpty():
        cur_state, cur_solu = fringe.pop()
        
        if not cur_state in closed_set:
            closed_set.add(cur_state)
        else:
            continue
        if problem.isGoalState(cur_state):
            return cur_solu

        succs = problem.getSuccessors(cur_state)
        for succ_state in succs:
            if succ_state not in closed_set:
                upd_solu = cur_solu + [succ_state[1]]
                succ_cost = problem.getCostOfActions(upd_solu) + heuristic(succ_state[0], problem)  # update A* cost
                fringe.update((succ_state[0], upd_solu), succ_cost)     # update fringes 




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
