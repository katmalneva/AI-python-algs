"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    frontier = Stack()
    visited = set()
    result = None
    frontier.push((problem.startingState(), [], 0))

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        if problem.isGoal(state):
            return path

        elif state not in visited:
            visited.add(state)

            for succState, succAction, succCost in problem.successorStates(state):
                if succState not in visited:
                    newPath = path + [succAction]
                    frontier.push((succState, newPath, succCost))

    return result


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    frontier = Queue()
    visited = set()
    result = None
    frontier.push((problem.startingState(), [], 0))

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        if problem.isGoal(state):
            return path

        elif state not in visited:
            visited.add(state)
            for succState, succAction, succCost in problem.successorStates(state):
                if succState not in visited:
                    newPath = path + [succAction]

                    frontier.push((succState, newPath, succCost))

    return result


def uniformCostSearch(problem):

    """
    Search the node of least total cost first.
    """
    frontier = PriorityQueue()
    visited = set()
    cost = 0
    frontier.push((problem.startingState(), [], 0), cost)

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        if problem.isGoal(state):
            return path

        elif state not in visited:
            visited.add(state)

            for succState, succAction, succCost in problem.successorStates(state):
                if succState not in visited:
                    newPath = path + [succAction]
                    frontier.push((succState, newPath, succCost), succCost)

    return None


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    frontier = PriorityQueue()
    visited = set()
    frontier.push((problem.startingState(), [], 0), heuristic(problem.startingState(), problem))

    while not frontier.isEmpty():

        state, path, cost = frontier.pop()

        if problem.isGoal(state):
            return path
        if state not in visited:
            visited.add(state)

            for succState, succAction, succCost in problem.successorStates(state):
                if succState not in visited:
                    newPath = path + [succAction]
                    f = heuristic(succState, problem) + succCost
                    frontier.push((succState, newPath, succCost), f)

    return None
