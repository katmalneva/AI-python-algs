from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions
from pacai.core.actions import Actions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.heuristic import null
from pacai.util import reflection
from pacai.util import util
from pacai.util import counter
from pacai.util import queue
from pacai.util import priorityQueue

import random
import copy

# information shared by teammates
# implement
IdentifiedInvaders = []
DistToInvader = []

two_invader_def =("",0)
twoInvanderDist = []



class BaseAgent(CaptureAgent):  # this is my base agent (mimic based on captureAgent and ReflexCaptureAgent)

    def registerInitialState(self, gameState):
        # overwrite parent method to populates more useful fields
        DistToInvader.clear()
        CaptureAgent.registerInitialState(self, gameState)
        self.startingPosition = gameState.getAgentPosition(self.index)
        self.height = gameState._layout.height
        self.width = gameState._layout.width
        self.walls = gameState.getWalls()
        self.borderline = gameState._layout.width / 2
        self.numOfFoodEaten = 0

        self.foodList = self.getFood(gameState).asList()
        self.safeEdge = self.getBorderline(gameState)
        self.totalFood = len(self.foodList)
        self.totalCapsule = len(self.getCapsules(gameState))

        foodWithOpening = self.getFoodList(gameState)
        self.safeFood = []

        for food in foodWithOpening:
            # print("food: ", food)
            numofWaysOut = self.numOfWaysOut(food)
            # print("numofWaysOut = ", numofWaysOut)

            if numofWaysOut > 1:
                self.safeFood.append(food[0])

        # print("self.safeFood length: ", len(self.safeFood))
        self.dangerFood = [food for food in self.foodList if food not in self.safeFood]
        self.totalSafeFood = len(self.safeFood)
        # print("self.safeFood: ", self.safeFood)
        # print("self.dangerFood  ", self.dangerFood

    def numOfWaysOut(self, foods):
        food, foodEntries = foods
        visited = []
        visited.append(food)
        count = 0
        for foodEntry in foodEntries:
            closed = copy.deepcopy(visited)
            reachHome = self.breadthFirstSearchSafeEdge(foodEntry, closed)
            # print("reachHome: ", reachHome)
            if reachHome:
                count = count + 1
        return count

    def getFoodList(self, gameState):
        foods = []
        # print("self.foodList: ", self.foodList)
        for food in self.foodList:
            openDirections = []
            count = 0
            FoodChain = []
            up = (food[0], food[1] + 1)
            down = (food[0], food[1] - 1)
            left = (food[0] - 1, food[1])
            right = (food[0] + 1, food[1])
            FoodChain.append(up)
            FoodChain.append(down)
            FoodChain.append(left)
            FoodChain.append(right)

            # print("FoodChain: ", FoodChain)
            for entry in FoodChain:
                if not self.walls[entry[0]][entry[1]]:
                    count = count + 1
                    openDirections.append(entry)
            if count > 1:
                foods.append((food, openDirections))
                # print("foods: ", foods)
        return foods




    def successorStates(self, state):
        """
        Returns successor states, the actions they require, and a constant cost of 1.
        """

        successors = []

        for action in Directions.CARDINAL:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if (not self.walls[nextx][nexty]):
                nextState = (nextx, nexty)

                successors.append((nextState, action))

        # # Bookkeeping for display purposes (the highlight in the GUI).
        # self._numExpanded += 1
        # if (state not in self._visitedLocations):
        #     self._visitedLocations.add(state)
        #     self._visitHistory.append(state)

        return successors

    def breadthFirstSearchFood(self, passed_fringe, visited):
        """
        Search the shallowest nodes in the search tree first. [p 81]
        """

        path = []
        fringe = queue.Queue()
        fringe.push((passed_fringe, path))

        while not fringe.isEmpty():

            cur_state, path = fringe.pop()

            if cur_state in self.foodList:
                return True
            else:
                for nei_state, nei_action in self.successorStates(cur_state):
                    if nei_state in visited:
                        continue
                    # if nei_state not in visited:
                    new_path = path + [nei_action]
                    fringe.push((nei_state, new_path))
                    visited.append(nei_state)

    def breadthFirstSearchSafeEdge(self, passed_fringe, visited):
        """
        Search the shallowest nodes in the search tree first. [p 81]
        """

        path = []
        fringe = queue.Queue()
        fringe.push((passed_fringe, path))

        while (not fringe.isEmpty()):

            cur_state, path = fringe.pop()

            if cur_state in self.safeEdge:
                return True
            else:
                for nei_state, nei_action in self.successorStates(cur_state):
                    if nei_state in visited:
                        continue
                    # if nei_state not in visited:
                    new_path = path + [nei_action]
                    fringe.push((nei_state, new_path))
                    visited.append(nei_state)

    def getBorderline(self, gameState):
        """middle line"""
        if self.red:
            Xpos = int(self.borderline - 1)
        else:
            Xpos = int(self.borderline)
        boudaries_withWall = [(Xpos, Ypos) for Ypos in range(self.height)]
        realBoudaries = []
        for location in boudaries_withWall:
            if not self.walls[location[0]][location[1]]:
                realBoudaries.append(location)
        # print("realBoudaries: ", realBoudaries)
        return realBoudaries

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """
        return {
            'successorScore': 1.0
        }

    def getBorderlineDist(self, gameState):

        cur_State = gameState.getAgentState(self.index)
        cur_Position = cur_State.getPosition()
        realBoudaries = self.getBorderline(gameState)
        realdist = float('inf')
        for location in realBoudaries:
            dist = self.getMazeDistance(location, cur_Position)
            realdist = min(realdist, dist)
        return realdist

    def distToNearestCapsule(self, gameState):

        cur_State = gameState.getAgentState(self.index)
        cur_Position = cur_State.getPosition()
        realdist = float('inf')
        for capsuleLocation in self.getCapsules(gameState):
            dist = self.getMazeDistance(cur_Position, capsuleLocation)
            realdist = min(realdist, dist)
        return realdist

    def distToNearestDefender(self, gameState):

        cur_Position = gameState.getAgentState(self.index).getPosition()
        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        defenders = [opponent for opponent in opponents if opponent.isGhost() and opponent.getPosition() is not None]
        # for a in opponents:
        #     print("a is pacman: ", a.isPacman())
        # print("cur_Position: ", cur_Position)

        if len(defenders) > 0:
            realdist = float('inf')
            realState = None
            for defender in defenders:
                dist = self.getMazeDistance(cur_Position, defender.getPosition())
                if dist < realdist:
                    realdist = dist
                    realState = defender
            return (realdist, realState)
        else:
            return None

    def distToNearestInvader(self, gameState):

        cur_Position = gameState.getAgentState(self.index).getPosition()
        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        invaders = [opponent for opponent in opponents if opponent.isPacman() and opponent.getPosition() is not None]

        if len(invaders) > 0:
            realdist = float('inf')
            for invader in invaders:
                dist = self.getMazeDistance(cur_Position, invader.getPosition())
                realdist = min(realdist, dist)
            return realdist
        else:
            return None

    def opponentScaredTimer(self, gameState):

        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        # for opponent in opponents:
        # print("opponent: ", opponents)
        # print("opponent.isPacman(): ", opponent.isPacman())
        # print("opponent.getScaredTimer(): ", opponent.getScaredTimer())
        scaredTimers = [opponent.getScaredTimer() for opponent in opponents]

        for scaredTimer in scaredTimers:
            # print("timer: ", opponent.getScaredTimer())
            if scaredTimer > 1:
                # print("scaredTimer: ", scaredTimer)
                return scaredTimer

        return None

    def aStarSearch(self, problem, gameState, heuristic=null):
        """
        Search the node that has the lowest combined cost and heuristic first.
        """

        path = []
        dist = {}
        dist[problem.startingState()[0]] = 0
        visited = []
        visited.append(problem.startingState())
        fringe = priorityQueue.PriorityQueue()
        fringe.push((problem.startingState(), path), dist[problem.startingState()[0]])
        # print("startstate: ", problem.startingState())

        while (fringe.__len__() != 0):
            cur_state, path = fringe.pop()

            if type(cur_state[0]) is tuple and hasattr(problem, 'goal'):
                if cur_state[0] in problem.goal:
                    cur_state[1].add(cur_state[0])

            if problem.isGoal(cur_state):
                # if self.index == 2:
                #     print("problem: ", problem)
                #     print("starting: ", problem.startingState())
                #     print(cur_state)
                # print("path: ", path)
                # print("---------------")
                return path
            else:

                for nei_state, nei_action, nei_cost in problem.successorStates(cur_state):
                    alt = dist[cur_state[0]] + nei_cost

                    # update the fringe if haven't visited or we find a shorter distance
                    if nei_state not in visited or alt < dist[nei_state[0]]:
                        visited.append(nei_state)
                        dist[nei_state[0]] = alt
                        new_path = path + [nei_action]
                        fringe.push((nei_state, new_path), alt + heuristic(nei_state, gameState))

    def CoopHeuristic(self, state, gameState):

        heuristic = 0

        if self.distToNearestDefender(gameState) is not None:
            opponents = [gameState.getAgentState(agent) for agent in self.getOpponents(gameState)]
            # pacmans = [a for a in opponents if a.isPacman()]
            # print("self indx: ")
            teammate = [agent for agent in self.getTeam(gameState) if agent != self.index][0]
            teammateLocation = gameState.getAgentState(teammate).getPosition()
            # print("state: ", state)
            # print("teammateLocation: ", teammateLocation)
            distToTeamate = self.getMazeDistance(state, teammateLocation)
            if distToTeamate == 0:
                heuristic = heuristic + 999
            else:
                heuristic = heuristic + (10 * (1 / distToTeamate))
            distToBorder = self.getBorderlineDist(gameState)
            heuristic = heuristic + distToBorder
            # print("distToTeamate: ", distToTeamate)
            # print("-----------")
            defenders = [defender for defender in opponents if defender.isGhost() and defender.getScaredTimer() < 2]
            if len(defenders) > 0:
                distToDefender = self.distToNearestDefender(gameState)[0]
                if distToDefender < 2:
                    heuristic = heuristic + (5 - distToDefender) ** 5
        # print("heuristic: ", heuristic)
        return heuristic


"""
Sheng's implementation of AStar
"""
class AstarTransform(BaseAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def ClosestOpponentBorderlineLocation(self, enemyLocation, gameState):

        realBoudaries = self.getBorderline(gameState)
        realdist = float('inf')
        closestLocation = None
        for location in realBoudaries:
            dist = self.getMazeDistance(location, enemyLocation)
            if dist < realdist:
                realdist = dist
                closestLocation = location

        return (closestLocation, realdist)

    def chooseAction(self, gameState):

        cur_State = gameState.getAgentState(self.index)
        cur_Position = cur_State.getPosition()
        cur_Score = self.getScore(gameState)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]

        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        defenders = [defender for defender in opponents if defender.isGhost()]
        invaders = [invader for invader in opponents if invader.isPacman()]
        foodList = self.getFood(gameState).asList()
        opponentTimer = self.opponentScaredTimer(gameState)
        capsuleList = self.getCapsules(gameState)
        distToDefender = self.distToNearestDefender(gameState)
        # print("getScaredTimer: ",getScaredTimer)

        # if gameState.getTimeleft() > 1100:
        if True:
            # if no invader we focus on attacking
            if len(invaders) < 1:
                if len(self.safeFood) < 1 and len(capsuleList) != 0 and opponentTimer is not None:
                    if opponentTimer < 10:
                        problem = SearchCapsuleProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if (len(self.safeFood) > 0):
                    problem = SearchSafeFoodProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    if len(path) > 0:
                        return path[0]

                if (len(self.safeFood) == 0):
                    problem = SearchFoodProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    return path[0]

                if distToDefender is not None:
                    dist = distToDefender[0]
                    time = distToDefender[1].getScaredTimer()
                    if dist < 5 and time < 5:

                        problem = EscapeToBorderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]
                        else:
                            return 'Stop'

                # print("self.opponentScaredTimer(gameState): ", self.opponentScaredTimer(gameState))
                if opponentTimer is not None:
                    if opponentTimer > (40 / 2) and len(self.dangerFood) > 0:
                        problem = SearchFoodInDangerCornerProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if len(foodList) < 3 or gameState.getTimeleft() < self.getBorderlineDist(gameState) + 200:
                    problem = EscapeToBorderProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    if len(self.aStarSearch(problem, self.CoopHeuristic)) == 0:
                        return path[0]
                    else:
                        return 'Stop'

                problem = SearchFoodProblem(gameState, self)
                path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                return path[0]


            elif len(invaders) >0:
                # print(self.index)
                # print("DistToInvader: ", DistToInvader)
                # print("_________________")
                distToBoth = []
                for invader in invaders:
                    distToBoth.append(self.getMazeDistance(invader.getPosition(), cur_Position))
                myDistToInvader = min(distToBoth)
                teammateDistToInvader = float('inf')
                if len(DistToInvader) == 0:
                    DistToInvader.append(myDistToInvader)
                else:
                    teammateDistToInvader = DistToInvader[-1]
                    DistToInvader.append(myDistToInvader)

                # when there are invader and I am closer to the invader, we excute defendense strategy
                if myDistToInvader < teammateDistToInvader:
                    if gameState.getAgentState(self.index).getScaredTimer() == 0:
                        # print("cur Postion: ", cur_Position)
                        problem = SearchInvaderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]
                else:  # otherwise I will keep attacking
                    if len(self.safeFood) < 1 and len(capsuleList) != 0 and opponentTimer is not None:
                        if opponentTimer < 10:
                            problem = SearchCapsuleProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            return path[0]

                    if (len(self.safeFood) > 0):
                        problem = SearchSafeFoodProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                    if (len(self.safeFood) == 0):
                        problem = SearchFoodProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                    if distToDefender is not None:
                        dist = distToDefender[0]
                        time = distToDefender[1].getScaredTimer()
                        if dist < 5 and time < 5:

                            problem = EscapeToBorderProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            if len(path) > 0:
                                return path[0]
                            else:
                                return 'Stop'

                    # print("self.opponentScaredTimer(gameState): ", self.opponentScaredTimer(gameState))
                    if opponentTimer is not None:
                        if opponentTimer > (40 / 2) and len(self.dangerFood) > 0:
                            problem = SearchFoodInDangerCornerProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            return path[0]

                    if len(foodList) < 3 or gameState.getTimeleft() < self.getBorderlineDist(gameState) + 200:
                        problem = EscapeToBorderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(self.aStarSearch(problem, self.CoopHeuristic)) == 0:
                            return path[0]
                        else:
                            return 'Stop'

            else:
                if gameState.getAgentState(self.index).getScaredTimer() == 0:
                    # print("cur Postion: ", cur_Position)
                    problem = SearchInvaderProblem(gameState, self)
                    #self.findAgentClosestToInvader(invaders, gameState)
                    return self.aStarSearch(problem, gameState, self.CoopHeuristic)[0]

                problem = SearchFoodProblem(gameState, self)
                path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                return path[0]

            # print(len(invaders))

            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)

    """def findAgentClosestToInvader(self, invaders, gameState):
        global two_invader_def
        cur_State = gameState.getAgentState(self.index)
        cur_Position = cur_State.getPosition()

        agentLocations = list()
        agentLocations.append(cur_Position)
        agentLocations.append(self.distToNearestDefender(gameState)[1].getPosition())

        dist_from_defenders_to_invaders = priorityQueue.PriorityQueue()
        for agent in agentLocations:
            for invader in invaders:
                mazeDist = self.getMazeDistance(invader.getPosition(), agent)
                dist_from_defenders_to_invaders.push((agent, invader, mazeDist), mazeDist)

        minDistOne = dist_from_defenders_to_invaders.pop()
        minDistTwo = dist_from_defenders_to_invaders.pop()
        if two_invader_def[0] != self.index:
            #case 1 if each agent is close one of the invaders
            if minDistOne[0] != minDistTwo[0] and minDistOne[1] != minDistTwo[1]:
                if minDistOne[0] == cur_Position or minDistTwo[0] == cur_Position:
                    problem = SearchInvaderProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    return path[0]


                else:
                    if minDistOne[0] != cur_Position:
                        two_invader_def = (self.index,minDistOne[2])
                    else:
                        two_invader_def = (self.index, minDistTwo[2])


            # case 2 if one agent is closer to both invaders
            if minDistOne[0] == minDistTwo[0] and minDistOne[1] != minDistTwo[1]:
                if minDistOne[0] == cur_Position:
                    # if our current agent is closer to both food

                else:
                    # if the other agent is closer to both foods

            #case 3 both our agents are closer to one invader
            if minDistOne[1] == minDistTwo[1] and minDistOne[0] == minDistTwo[0]:
                if minDistOne[0] == cur_Position:


            else:
                #catch fallout
        else:
            """






    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # features['dead'] = 0

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in opponents if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0 and gameState.getAgentState(self.index).getScaredTimer() > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = 10 * (-1 / min(dists))

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        if myState.isScared():
            features['scared'] = 1

        features['DistToBoundary'] = - self.getBorderlineDist(successor)

        if self.red:
            if myPos == (1, 1):
                features['death'] = 1
        else:
            if myPos == (self.height - 1, self.width - 1):
                features['death'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'onDefense': 200,
            'invaderDistance': -10,
            'stop': -200,
            'reverse': -5,
            'DistToBoundary': 3,
            'death': -9999,
            'scared': -30
        }


class SearchFoodChain(PositionSearchProblem):

    def __init__(self, gameState, agent):

        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.foodChain = agent.foodChain
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.foodLeft = len(self.foodList)

    def isGoal(self, state):
        return state in self.foodChain


class SearchFoodProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.foodLeft = len(self.foodList)

    def isGoal(self, state):
        return state in self.foodList


class SearchSafeFoodProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.foodLeft = len(self.foodList)
        self.safeFood = agent.safeFood

    def isGoal(self, state):
        return state in self.safeFood


class SearchFoodInDangerCornerProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.foodLeft = len(self.foodList)
        self.dangerousFood = agent.dangerFood

    def isGoal(self, state):
        return state in self.dangerousFood


class DefendBorderANDLockOP1Problem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.safeEdge = agent.getBorderline(gameState)
        self.opponents = [gameState.getAgentState(opponent) for opponent in agent.getOpponents(gameState)]
        self.defenders = [defender for defender in self.opponents if defender.isGhost()]
        if len(self.defenders) > 0:
            self.defenderLocations = [defender.getPosition() for defender in self.defenders]
        else:
            self.defenderLocations = None

    def startingState(self):
        return self.startState

    def isGoal(self, state):
        return state in self.defenderLocations


class EscapeToBorderProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.safeEdge = agent.getBorderline(gameState)

    def startingState(self):
        return self.startState

    def isGoal(self, state):
        return (state in self.safeEdge or state in self.capsule)


class SearchCapsuleProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1

    def isGoal(self, state):
        return state in self.capsule


class SearchInvaderProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        super().__init__(gameState)
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.opponents = [gameState.getAgentState(opponent) for opponent in agent.getOpponents(gameState)]
        self.invaders = [a for a in self.opponents if a.isPacman()]
        if len(self.invaders) > 0:
            self.invaderLocations = [invader.getPosition() for invader in self.invaders]
        else:
            self.invaderLocations = None

    def isGoal(self, state):
        return state in self.invaderLocations


