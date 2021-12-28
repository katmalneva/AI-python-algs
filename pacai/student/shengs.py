from pacai.agents.capture.capture import CaptureAgent
from pacai.core import distance
from pacai.core.directions import Directions
from pacai.core.actions import Actions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.heuristic import null
from pacai.util import reflection, util, counter, queue, priorityQueue
from pacai.core.distance import manhattan

import random
import copy

# information shared by teammates
# implement
IdentifiedInvaders = []
DistToInvader = []
DistToFood = []
FoodHunting = [None]
GoalList = {}

visited = []

def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.student.myTeam.AstarTransform',
               second='pacai.student.myTeam.AstarTransform'):
    # first = 'pacai.agents.capture.offense.OffensiveReflexAgent',
    # second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = AstarTransform
    secondAgent = AstarTransform

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]


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
        GoalList[self.index] = []

        self.foodList = self.getFood(gameState).asList()
        self.foodProtectedList = self.getFoodYouAreDefending(gameState).asList()
        self.safeEdge = self.getBorderline(gameState)
        self.totalFood = len(self.foodList)
        self.totalCapsule = len(self.getCapsules(gameState))
        self.foodChain = {}

        # initialize the foodchain graph
        # for food in self.foodList:
        #     self.

        foodWithOpening = self.getFoodList(gameState)

        opponentfoodWithOpening = self.getFoodProtectingList(gameState)

        # print("self.foodWithOpening: ", foodWithOpening)
        # print("self.foodChain: ", self.foodChain)
        foodChain_copy = copy.deepcopy(self.foodChain)
        # print("foodlist: ", self.foodList)
        # print("____________")
        # print("opponentfoodWithOpening: ", opponentfoodWithOpening)

        for food, entries in foodChain_copy.items():
            for entry in entries:
                # print("entry: ", entry)
                if entry not in self.foodList:
                    # print("deleted entry: ", entry)
                    self.foodChain[food].remove(entry)
            # print("-------------------")

        # print("self.foodChain: ", self.foodChain)

        def getRoots(foodChain):
            def findRoot(aNode, aRoot):
                while aNode != aRoot[aNode][0]:
                    aNode = aRoot[aNode][0]

                return (aNode, aRoot[aNode][1])

            myRoot = {}
            for food in foodChain.keys():
                myRoot[food] = (food, 0)

            # print("myRoot: ", myRoot)
            for food in foodChain:
                for adjfood in foodChain[food]:
                    (myRoot_myI, myDepthMyI) = findRoot(food, myRoot)
                    (myRoot_myJ, myDepthMyJ) = findRoot(adjfood, myRoot)
                    if myRoot_myI != myRoot_myJ:
                        myMin = myRoot_myI
                        myMax = myRoot_myJ
                        if myDepthMyI > myDepthMyJ:
                            myMin = myRoot_myJ
                            myMax = myRoot_myI
                        myRoot[myMax] = (myMax, max(myRoot[myMin][1] + 1, myRoot[myMax][1]))
                        myRoot[myMin] = (myRoot[myMax][0], -1)
            myToRet = {}
            for food in foodChain:
                if myRoot[food][0] == food:
                    myToRet[food] = []
            for food in foodChain:
                # print("findRoot(food,myRoot)[0]: ", findRoot(food,myRoot)[0])
                myToRet[findRoot(food, myRoot)[0]].append(food)
            return myToRet

        self.realFoodChain = getRoots(self.foodChain)
        # for root, foodchain in self.realFoodChain.items():
        #     if foodChain
        # print("foodWithOpening: ", foodWithOpening)
        # print("self.realFoodChain: ", self.realFoodChain)

        self.safeFood = []

        for food in foodWithOpening:
            # print("food: ", food)
            numofWaysOut = self.numofWaysOut(food)
            # print("numofWaysOut = ", numofWaysOut)

            if numofWaysOut > 1:
                self.safeFood.append(food[0])

        self.opponentSafeFood = []

        for food in opponentfoodWithOpening:
            # print("food: ", food)
            numofWaysOut = self.numofWaysOut(food)
            # print("numofWaysOut = ", numofWaysOut)

            if numofWaysOut > 1:
                self.opponentSafeFood.append(food[0])

        # print("self.safeFood length: ", len(self.safeFood))
        self.dangerFood = [food for food in self.foodList if food not in self.safeFood]
        self.totalSafeFood = len(self.safeFood)
        # print("self.safeFood: ", self.safeFood)
        # print("self.opponentSafeFood: ", self.opponentSafeFood)
        # print("self.dangerFood  ", self.dangerFood

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
            self.foodChain[food] = FoodChain

            # print("FoodChain: ", FoodChain)
            for entry in FoodChain:
                if not self.walls[entry[0]][entry[1]]:
                    count = count + 1
                    openDirections.append(entry)
            if count > 1:
                # self.foodChain[food] = openDirections
                foods.append((food, openDirections))
                # print("foods: ", foods)
        return foods

    def getFoodProtectingList(self, gameState):
        foods = []
        # print("self.foodList: ", self.foodList)
        for food in self.foodProtectedList:
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

    def numofWaysOut(self, foods):
        # print("passed foods: ", foods)
        food, foodEntries = foods
        visited = []
        visited.append(food)
        count = 0
        for foodEntry in foodEntries:
            visitedCopy = copy.deepcopy(visited)
            reachHome = self.breadthFirstSearchSafeEdge(foodEntry, visitedCopy)

            # if self.index == 0:
            # print(self.index)
            # print("food: ", food)
            # print("foodEntry: ", foodEntry)
            # print("reachHome: ", reachHome)

            if reachHome:
                count = count + 1
        # print("----------------------")
        return count

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

        while (not fringe.isEmpty()):

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

        # *** Your Code Here ***
        # if self.index == 0:
        #     print(problem)
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

            if problem.isGoal(cur_state):
                # if self.index == 0:
                # print(self.index)
                # print("problem: ", problem)
                # print("cur_state: ", cur_state)
                #     print("starting: ", problem.startingState())
                # print(cur_state)
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
        cur_Position = state

        if self.distToNearestDefender(gameState) is not None:
            opponents = [gameState.getAgentState(agent) for agent in self.getOpponents(gameState)]
            # pacmans = [a for a in opponents if a.isPacman()]
            # print("self indx: ")
            teammate = [agent for agent in self.getTeam(gameState) if agent != self.index][0]
            teammateLocation = gameState.getAgentState(teammate).getPosition()
            # print("cur_Position: ", cur_Position)
            # print("teammateLocation: ", teammateLocation)

            # if self.red:
            #     Xpos = int(self.borderline + 1)
            #     if cur_Position[0] >= Xpos:
            #     # distance to teammates
            #         distToTeamate = self.getMazeDistance(cur_Position, teammateLocation)
            #         if distToTeamate == 0:
            #             heuristic = heuristic + 99
            #         else:
            #             heuristic = heuristic + (50 * (1 / distToTeamate))
            # else:
            #     Xpos = int(self.borderline - 2)
            #     if cur_Position[0] <= Xpos:
            #     # distance to teammates
            #         distToTeamate = self.getMazeDistance(cur_Position, teammateLocation)
            #         if distToTeamate == 0:
            #             heuristic = heuristic + 99
            #         else:
            #             heuristic = heuristic + (50 * (1 / distToTeamate))

            distToTeamate = self.getMazeDistance(cur_Position, teammateLocation)
            if distToTeamate == 0:
                heuristic = heuristic + 999
            else:
                heuristic = heuristic + (50 * (1 / distToTeamate))

            distToBorder = self.getBorderlineDist(gameState)
            heuristic = heuristic + distToBorder
            # print("distToTeamate: ", distToTeamate)
            # print("-----------")
            # for defender in opponents:
            #     if defender.isGhost() and defender.getScaredTimer() < 2:
            #         print("yesssss")
            defenders = [defender for defender in opponents if defender.isGhost() and defender.getScaredTimer() < 2]
            if len(defenders) > 0:
                defenderLocations = [defender.getPosition() for defender in defenders]
                realdist = float('inf')
                for location in defenderLocations:
                    dist = self.getMazeDistance(location, cur_Position)
                    realdist = min(realdist, dist)
                if realdist < 2:
                    if realdist == 0:
                        heuristic = heuristic + 9999
                    else:
                        heuristic = heuristic + (999 * (1 / realdist))

        # if self.index == 0:
        #     print("heuristic: ", heuristic)
        return heuristic

    def HuntingHeuristic(self, state, gameState):

        heuristic = 0
        cur_Position = state

        if self.distToNearestDefender(gameState) is not None:
            opponents = [gameState.getAgentState(agent) for agent in self.getOpponents(gameState)]
            # pacmans = [a for a in opponents if a.isPacman()]
            # print("self indx: ")
            teammate = [agent for agent in self.getTeam(gameState) if agent != self.index][0]
            teammateLocation = gameState.getAgentState(teammate).getPosition()
            # print("cur_Position: ", cur_Position)
            # print("teammateLocation: ", teammateLocation)

            # if self.red:
            #     Xpos = int(self.borderline + 1)
            #     if cur_Position[0] >= Xpos:
            #     # distance to teammates
            #         distToTeamate = self.getMazeDistance(cur_Position, teammateLocation)
            #         if distToTeamate == 0:
            #             heuristic = heuristic + 99
            #         else:
            #             heuristic = heuristic + (50 * (1 / distToTeamate))
            # else:
            #     Xpos = int(self.borderline - 2)
            #     if cur_Position[0] <= Xpos:
            #     # distance to teammates
            #         distToTeamate = self.getMazeDistance(cur_Position, teammateLocation)
            #         if distToTeamate == 0:
            #             heuristic = heuristic + 99
            #         else:
            #             heuristic = heuristic + (50 * (1 / distToTeamate))

            distToTeamate = self.getMazeDistance(cur_Position, teammateLocation)
            if distToTeamate == 0:
                heuristic = heuristic + 999
            else:
                heuristic = heuristic + (50 * (1 / distToTeamate))

            distToBorder = self.getBorderlineDist(gameState)
            heuristic = heuristic + distToBorder
            # print("distToTeamate: ", distToTeamate)
            # print("-----------")
            # for defender in opponents:
            #     if defender.isGhost() and defender.getScaredTimer() < 2:
            #         print("yesssss")
            defenders = [defender for defender in opponents if defender.isGhost() and defender.getScaredTimer() < 2]
            if len(defenders) > 0:
                defenderLocations = [defender.getPosition() for defender in defenders]
                realdist = float('inf')
                for location in defenderLocations:
                    dist = self.getMazeDistance(location, cur_Position)
                    realdist = min(realdist, dist)
                if realdist < 2:
                    if realdist == 0:
                        heuristic = heuristic + 9999
                    else:
                        heuristic = heuristic + (999 * (1 / realdist))

        # if self.index == 0:
        #     print("heuristic: ", heuristic)
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
        global visited
        cur_State = gameState.getAgentState(self.index)
        cur_Position = cur_State.getPosition()
        cur_Score = self.getScore(gameState)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]

        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        defenders = [defender for defender in opponents if defender.isGhost()]
        invaders = [invader for invader in opponents if invader.isPacman()]
        foodList = self.getFood(gameState).asList()
        opponentFoodList = self.getFoodYouAreDefending(gameState).asList()

        updatedGoalList = [food for food in GoalList[self.index] if food in foodList]
        GoalList[self.index] = copy.deepcopy(updatedGoalList)
        # print("GoalList[self.index]: ", GoalList[self.index])
        # updatedFoodChain = {food:adjfood for food, adjfood in self.realFoodChain.items() if food in foodList}
        updatedFoodChain = {}
        for food, adjfood in self.realFoodChain.items():
            tempList = copy.deepcopy(adjfood)
            for adj in adjfood:
                if adj not in foodList:
                    tempList.remove(adj)
            updatedFoodChain[food] = tempList

        updatedFoodChain_new = {food: adjfood for food, adjfood in updatedFoodChain.items() if food in foodList}

        self.realFoodChain = copy.deepcopy(updatedFoodChain_new)

        # print("self.realFoodChain: ", self.realFoodChain)
        # print("GoalList[self.index]: ", GoalList[self.index])
        # print(self.borderline)
        # print(self.getFood(gameState))
        # print("foodList: ", foodList)
        # print("=============")
        updatedOpponentSafeFood = [food for food in opponentFoodList if food in self.opponentSafeFood]
        updatedSafeFood = [food for food in foodList if food in self.safeFood]
        updatedDangerFood = [food for food in foodList if food in self.dangerFood]
        self.opponentSafeFood = copy.deepcopy(updatedOpponentSafeFood)
        self.safeFood = copy.deepcopy(updatedSafeFood)
        self.dangerFood = copy.deepcopy(updatedDangerFood)
        protectFoodList = self.getFoodYouAreDefending(gameState).asList()
        opponentTimer = self.opponentScaredTimer(gameState)
        capsuleList = self.getCapsules(gameState)
        distToDefender = self.distToNearestDefender(gameState)
        distToInvader = self.distToNearestInvader(gameState)
        distToBorder = self.getBorderlineDist(gameState)
        isBraveDefender = opponentTimer is None

        # if gameState.getTimeleft() > 500:
        if self.totalSafeFood == 0 and self.totalCapsule == 1:

            # if no invaders we focus on attacking
            if len(invaders) < 1:
                if len(self.safeFood) < 1 and len(capsuleList) != 0 and (not isBraveDefender):
                    if opponentTimer < 10:
                        FoodHunting.append(None)
                        problem = SearchCapsuleProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if (len(self.safeFood) > 0 and isBraveDefender and distToDefender is not None):
                    if distToDefender[0] > 2:
                        GoalList[self.index].clear()
                        problem = SearchSafeFoodProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                if (len(self.safeFood) == 0 and isBraveDefender and distToDefender is not None):
                    # print("First anyfoodproblem")
                    if distToDefender[0] > 2:
                        problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                # print(len(self.safeFood))
                # print("here")

                if distToDefender is not None:
                    dist = distToDefender[0]
                    time = distToDefender[1].getScaredTimer()
                    if dist < 5 and time < 5:
                        FoodHunting.append(None)
                        problem = EscapeToBorderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]
                        else:
                            return 'Stop'

                # print("self.opponentScaredTimer(gameState): ", self.opponentScaredTimer(gameState))
                if not isBraveDefender:
                    if opponentTimer > (40 / 2) and len(self.dangerFood) > 0:
                        GoalList[self.index].clear()
                        problem = SearchFoodInDangerCornerProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                if len(foodList) > 3 and len(protectFoodList) < 4 and gameState.getTimeleft() < self.getBorderlineDist(
                        gameState) + 200:
                    FoodHunting.append(None)
                    problem = EscapeToBorderProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    if len(self.aStarSearch(problem, self.CoopHeuristic)) == 0:
                        return path[0]
                    else:
                        return 'Stop'
                # if self.index == 0:
                #     print("latter foodproblem")
                problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                return path[0]

            else:
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

                # FoodHunting.append(None)
                # subProblem = AnyFoodProblem(gameState, self)
                # subpath = self.aStarSearch(subProblem, gameState, self.CoopHeuristic)
                # myDistToFood = len(subpath)
                # teammateDistToFood = float('inf')
                # if len(DistToFood) == 0:
                #     DistToFood.append(myDistToFood)
                # else:
                #     teammateDistToFood = DistToFood[-1]
                #     DistToFood.append(myDistToFood)

                # when there are invader and I am closer to the invader, we excute defendense strategy
                if myDistToInvader < teammateDistToInvader and len(foodList) > 4 and gameState.getAgentState(
                        self.index).getScaredTimer() == 0:
                    # further idea: showdown issue
                    # if len(foodList) < 4 and len(protectFoodList) < 4:

                    #     opponentLocations = [opponent.getPosition() for opponent in opponents]
                    #     minDistOpponent = float('inf')
                    #     for location in opponentLocations:
                    #         for food in protectFoodList:
                    #             dist = self.getMazeDistance(location, food)
                    #             minDistOpponent = min(minDist, dist)

                    #     if myDistToFood <= minDistOpponent and myDistToFood <= teammateDistToFood:
                    #         return subpath[0]
                    #     else:
                    # if gameState.getAgentState(self.index).getScaredTimer() == 0:
                    FoodHunting.append(None)
                    problem = SearchInvaderProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    return path[0]

                else:  # otherwise I will keep attacking
                    if len(self.safeFood) < 1 and len(capsuleList) != 0 and (not isBraveDefender):
                        if opponentTimer < 10:
                            FoodHunting.append(None)
                            problem = SearchCapsuleProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            return path[0]

                    if (len(self.safeFood) > 0 and isBraveDefender and distToDefender is not None):
                        if distToDefender[0] > 2:
                            GoalList[self.index].clear()
                            problem = SearchSafeFoodProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            if len(path) > 0:
                                return path[0]

                    if (len(self.safeFood) == 0 and isBraveDefender and distToDefender is not None):
                        # print("First anyfoodproblem")
                        if distToDefender[0] > 2:
                            problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            return path[0]

                    if distToDefender is not None:
                        dist = distToDefender[0]
                        time = distToDefender[1].getScaredTimer()
                        if dist < 5 and time < 5:
                            FoodHunting.append(None)
                            problem = EscapeToBorderProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            if len(path) > 0:
                                return path[0]
                            else:
                                return 'Stop'

                    # print("self.opponentScaredTimer(gameState): ", self.opponentScaredTimer(gameState))
                    if not isBraveDefender:
                        if opponentTimer > (40 / 2) and len(self.dangerFood) > 0:
                            GoalList[self.index].clear()
                            problem = SearchFoodInDangerCornerProblem(gameState, self)
                            path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                            if len(path) > 0:
                                return path[0]

                    if len(foodList) > 3 and len(
                            protectFoodList) < 4 and gameState.getTimeleft() < self.getBorderlineDist(gameState) + 100:
                        FoodHunting.append(None)
                        problem = EscapeToBorderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(self.aStarSearch(problem, self.CoopHeuristic)) == 0:
                            return path[0]
                        else:
                            return 'Stop'

                    problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    return path[0]

        else:  # not extreme map
            if len(invaders) < 1:
                # print("HERERERERERERER____________")
                if len(self.safeFood) < 1 and len(capsuleList) != 0 and (not isBraveDefender):
                    if opponentTimer < 10:
                        FoodHunting.append(None)
                        problem = SearchCapsuleProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if (len(self.safeFood) > 0 and isBraveDefender and distToDefender is not None):
                    if distToDefender[0] > 2:
                        GoalList[self.index].clear()
                        problem = SearchSafeFoodProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                if (len(self.safeFood) == 0 and isBraveDefender):
                    # print("First anyfoodproblem")
                    if distToDefender[0] > 2:
                        problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if distToDefender is not None:
                    dist = distToDefender[0]
                    time = distToDefender[1].getScaredTimer()
                    if dist < 5 and time < 5:
                        FoodHunting.append(None)
                        problem = EscapeToBorderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]
                        else:
                            return 'Stop'

                # print("self.opponentScaredTimer(gameState): ", self.opponentScaredTimer(gameState))
                if not isBraveDefender:
                    if opponentTimer > (40 / 2) and len(self.dangerFood) > 0:
                        GoalList[self.index].clear()
                        problem = SearchFoodInDangerCornerProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                if len(foodList) > 3 and len(protectFoodList) < 4 and gameState.getTimeleft() < self.getBorderlineDist(
                        gameState) + 200:
                    FoodHunting.append(None)
                    problem = EscapeToBorderProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    if len(self.aStarSearch(problem, self.CoopHeuristic)) == 0:
                        return path[0]
                    else:
                        return 'Stop'
                # if self.index == 0:
                #     print("latter foodproblem")

                problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                return path[0]

            elif len(invaders) == 1:
                if distToInvader is not None and gameState.getAgentState(self.index).getScaredTimer() == 0:
                    if distToInvader < 6:
                        FoodHunting.append(None)
                        problem = SearchInvaderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if len(self.safeFood) < 1 and len(capsuleList) != 0 and (not isBraveDefender):
                    if opponentTimer < 10:
                        FoodHunting.append(None)
                        problem = SearchCapsuleProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if (len(self.safeFood) > 0 and isBraveDefender):
                    if distToDefender[0] > 2:
                        GoalList[self.index].clear()
                        problem = SearchSafeFoodProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                if (len(self.safeFood) == 0 and isBraveDefender):
                    # print("First anyfoodproblem")
                    if distToDefender[0] > 2:
                        problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                if distToDefender is not None:
                    dist = distToDefender[0]
                    time = distToDefender[1].getScaredTimer()
                    if dist < 5 and time < 5:
                        FoodHunting.append(None)
                        problem = EscapeToBorderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]
                        else:
                            return 'Stop'

                # print("self.opponentScaredTimer(gameState): ", self.opponentScaredTimer(gameState))
                if not isBraveDefender:
                    if opponentTimer > (40 / 2) and len(self.dangerFood) > 0:
                        GoalList[self.index].clear()
                        problem = SearchFoodInDangerCornerProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        if len(path) > 0:
                            return path[0]

                if len(foodList) > 3 and len(protectFoodList) < 4 and gameState.getTimeleft() < self.getBorderlineDist(
                        gameState) + 200:
                    FoodHunting.append(None)
                    problem = EscapeToBorderProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    if len(self.aStarSearch(problem, self.CoopHeuristic)) == 0:
                        return path[0]
                    else:
                        return 'Stop'

                problem = AnyFoodProblem(gameState, self, cur_Position, foodList)
                path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                return path[0]

            else:  # two invaders
                if distToInvader is not None and gameState.getAgentState(self.index).getScaredTimer() == 0:
                    if distToInvader < 3:
                        FoodHunting.append(None)
                        # FoodHunting.append(None)
                        problem = SearchInvaderProblem(gameState, self)
                        path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                        return path[0]

                # problem = AnyFoodProblem(gameState, self)
                # path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                # return path[0]
                if len(GoalList[self.index]) == 0:
                    problem = SearchFoodChainProblem(gameState, self)
                    path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                    # print("path: ", path)
                    if path is not None and len(path) > 0:
                        return path[0]

                problem = AnyFoodProblem(gameState, self, cur_Position, GoalList[self.index])
                path = self.aStarSearch(problem, gameState, self.CoopHeuristic)
                # print("path: ", path)
                if path is not None and len(path) > 0:
                    return path[0]

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
            # Only half a grid position was covered.
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
        global visited
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['visited'] = 0
        if myState in visited:
            features['visited'] = 1
            #print("is in visited")
            if len(visited) > 15:
                visited.clear()
        else:
            visited.append(myState)

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        features['onOffense'] = 0
        if (myState.isPacman()):
            features['onDefense'] = 0
            features['onOffense'] = 1

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        features["nearDeath"] = 0
        features["avoidGhost"] = 0
        distAgentToEnemy = []
        if myState.isPacman():
            for opponent in enemies:
                dist = 99999
                if opponent.isBraveGhost():
                    dist = self.getMazeDistance(myPos, opponent.getPosition())
                    distAgentToEnemy.append(dist)
                if dist <= 2 and opponent.isBraveGhost():
                    features["nearDeath"] = 1
                if len(distAgentToEnemy) > 0:
                    features["avoidGhost"] = 1 / min(distAgentToEnemy)

        features["killOpponent"] = 0
        distAgentToScaredGhost = []
        if myState.isPacman():
            for opponent in enemies:
                dist = self.getMazeDistance(myPos, opponent.getPosition())
                distAgentToScaredGhost.append(dist)
                if dist <= 4 and opponent.isScared():
                    features["killOpponent"] = 1 / min(distAgentToScaredGhost)

        features['attack'] = 1
        distAgentToInv = []
        if myState.isBraveGhost():
            for opponent in invaders:
                dist = self.getMazeDistance(myPos, opponent.getPosition())
                distAgentToInv.append(dist)
                features['attack'] = 1 / min(distAgentToInv)

        foodDistList = []
        oldFood = self.getFood(gameState)
        features['isFood'] = 0
        features["closestFood"] = 0
        for food in oldFood.asList():
            foodDist = self.getMazeDistance(myPos, food)
            foodDistList.append(foodDist)

        if len(foodDistList) > 0:
            closestFood = min(foodDistList)
            if closestFood == 0:
                features['isFood'] = 1
            else:
                features["closestFood"] = 1 / closestFood

        features['scared'] = 0
        distScaredAgentToInv = []
        if myState.isScared():
            for opponent in invaders:
                dist = self.getMazeDistance(myPos, opponent.getPosition())
                distScaredAgentToInv.append(dist)
                if min(distScaredAgentToInv) <= 3 and opponent.isPacman():
                    features['scared'] = 1 / min(distScaredAgentToInv)

        features['successorScore'] = self.getScore(successor)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -80,
            'onDefense': 9,
            'invaderDistance': 1,
            'stop': -100,
            'reverse': -10,
            'closestFood': 30,
            'isFood': 76,
            'onOffense': 16,
            "nearDeath": -1000,
            'scared': -377,
            'visited': -6,
            "avoidGhost": -7.5,
            'killOpponent': 8,
            'successorScore': 1,
            'attack': 90

        }

""""    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        foodDistList = []
        oldFood = self.getFood(gameState)

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onOffense'] = 1
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in opponents if a.isPacman()]
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

        features['successorScore'] = self.getScore(successor)

        # adapting name for use from previous pa
        currState = gameState

        # extracting all info
        succState = self.getSuccessor(currState, action)
        succPosition = succState.getAgentState(self.index).getPosition()

        currFood = self.getFood(currState).asList()
        succFood = self.getFood(succState).asList()

        succGStates = self.getOpponents(succState)
        self.getS
        features['noFoodLeft'] = 0
        if (len(succFood) <= 2):
            features['noFoodLeft'] = 100

        for food in oldFood.asList():
            foodDist = distance.manhattan(myState.getPosition(), food)
            foodDistList.append(foodDist)

        if len(foodDistList) > 0:
            closestFood = min(foodDistList)
            if closestFood == 0:
                features['isFood'] = 1
            else:
                features[closestFood] = 1 / closestFood

        features['ghost2Away'] = 0

        ghostList = [succState.getAgentState(ghost).getPosition() for ghost in succGStates]

        for ghost in ghostList:
            if self.getMazeDistance(succPosition, ghost) <= 2:
                features['ghost2Away'] = -1000

        nearestFood = float('inf')
        for food in succFood:
            currMazeLen = self.getMazeDistance(succPosition, food)
            nearestFood = min(nearestFood, currMazeLen)

        features['distToFoodInverse'] = 1 / nearestFood
        if (len(succFood) < len(currFood)):  # food was eaten
            features['distToFoodInverse'] += 3


        return features

    def getWeights(self, gameState, action):
        return {
            'onDefense': 10,
            'onOffense': 30,
            'invaderDistance': -10,
            'stop': -200,
            'reverse': -2,
            'DistToBoundary': 1,
            'death': -9999,
            'successorScore': 1,
            'noFoodLeft': 9999,
            'ghost2Away': 99,
            'distToFoodInverse': 1,
            'isFood': 5,
            'scared': -30
        }"""


class SearchFoodChainProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self.agent = agent
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.foodChain = agent.realFoodChain
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.goal = set()
        self.target = []
        self.costFn = lambda x: 1

    def startingState(self):

        return self.startState

    def isGoal(self, state):
        if FoodHunting[-1] is None:
            if state in self.foodChain:
                FoodHunting.append(state)
                GoalList[self.agent.index] = self.foodChain[state]
                # tempList = copy.deepcopy(self.foodChain[state])
                # tempList.remove(state)
                # print("self.foodChain[state]: ", self.foodChain[state])
                # print("tempList: ", tempList)
                self.goal.update(self.foodChain[state])
            # if len(self.goal) == 0:
            #     return False
            # return self.goal == state[1]
            return state in self.foodChain
        else:
            teammateFoodHunt = FoodHunting[-1]
            if state in self.foodChain and state != teammateFoodHunt:
                FoodHunting.append(state)
                GoalList[self.agent.index] = self.foodChain[state]
                tempList = copy.deepcopy(self.foodChain[state])
                tempList.remove(state)
                # print("state: ", state)
                # print("self.foodChain[state]: ", self.foodChain[state])
                # print("tempList: ", tempList)
                self.goal.update(tempList)
            # if len(self.goal) == 0:
            #     return False
            # return self.goal == state[1]
            return state in self.foodChain and state != teammateFoodHunt
        # return state in self.foodChain

    # def successorStates(self, state):
    #     """
    #     Returns successor states, the actions they require, and a constant cost of 1.
    #     """

    #     successors = []
    #     self._numExpanded += 1
    #     for action in Directions.CARDINAL:
    #         print("state: ", state)
    #         x, y = state[0]
    #         dx, dy = Actions.directionToVector(action)
    #         nextx, nexty = int(x + dx), int(y + dy)
    #         hitsWall = self.walls[nextx][nexty]

    #         if (not hitsWall):
    #             goalObtained = state[1].copy()
    #             nextState = ((nextx, nexty), goalObtained)
    #             cost = 1
    #             successors.append((nextState, action, cost))

    #     return successors


class AnyFoodProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, starting, food_list):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = food_list
        self.capsule = agent.getCapsules(gameState)
        self.startState = starting
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.goal = []

    def startingState(self):

        return self.startState

    def isGoal(self, state):

        if FoodHunting[-1] is None:
            if state in self.foodList:
                FoodHunting.append(state)
            return state in self.foodList
        else:
            teammateFoodHunt = FoodHunting[-1]
            if state in self.foodList and state != teammateFoodHunt:
                FoodHunting.append(state)
            return state in self.foodList and state != teammateFoodHunt
        # return state in self.foodList


class AnyTwoFoodProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, starting, food_list):
        self.gameState = gameState
        self.agent = agent
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = food_list
        self.capsule = agent.getCapsules(gameState)
        self.startState = starting
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.count = 0
        self.goal = []

    def startingState(self):

        return self.startState

    def isGoal(self, state):

        if FoodHunting[-1] is None:
            if state in self.foodList:
                FoodHunting.append(state)
                self.foodList.remove(state)
                problem = AnyFoodProblem(self.gameState, self.agent, state, self.foodList)
                path = self.agent.aStarSearch(problem, self.gameState, self.agent.CoopHeuristic)

            return state in self.foodList
        else:
            teammateFoodHunt1 = FoodHunting[-1]
            teammateFoodHunt2 = FoodHunting[-2]
            if state in self.foodList and state != teammateFoodHunt1 and state != teammateFoodHunt2:
                FoodHunting.append(state)
            return state in self.foodList and state != teammateFoodHunt1 and state != teammateFoodHunt2
        # return state in self.foodList


class SearchSafeFoodProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.safeFood = agent.safeFood
        self.goal = []

    def startingState(self):

        return self.startState

    def isGoal(self, state):

        if FoodHunting[-1] is None or len(self.safeFood) == 1:
            if state in self.safeFood:
                FoodHunting.append(state)
            return state in self.safeFood
        else:
            teammateFoodHunt = FoodHunting[-1]
            if state in self.safeFood and state != teammateFoodHunt:
                FoodHunting.append(state)
            return state in self.safeFood and state != teammateFoodHunt
        # return state in self.safeFood


class SearchFoodInDangerCornerProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.foodList = agent.getFood(gameState).asList()
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.dangerousFood = agent.dangerFood
        self.goal = []

    def startingState(self):

        return self.startState

    def isGoal(self, state):

        if FoodHunting[-1] is None or len(self.dangerousFood) == 1:
            if state in self.dangerousFood:
                FoodHunting.append(state)
            return state in self.dangerousFood
        else:
            teammateFoodHunt = FoodHunting[-1]
            if state in self.dangerousFood and state != teammateFoodHunt:
                FoodHunting.append(state)
            return state in self.dangerousFood and state != teammateFoodHunt
        # return state in self.dangerousFood


class EscapeToBorderProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.safeEdge = agent.getBorderline(gameState)
        self.goal = []

    def startingState(self):
        return self.startState

    def isGoal(self, state):
        return (state in self.capsule or state in self.safeEdge)


class SearchCapsuleProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.goal = []

    def startingState(self):
        return self.startState

    def isGoal(self, state):
        return state in self.capsule


class SearchInvaderProblem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.goal = []
        self.opponents = [gameState.getAgentState(opponent) for opponent in agent.getOpponents(gameState)]
        self.invaders = [a for a in self.opponents if a.isPacman()]
        if len(self.invaders) > 0:
            self.invaderLocations = [invader.getPosition() for invader in self.invaders]
        else:
            self.invaderLocations = None

    def startingState(self):

        return self.startState

    def isGoal(self, state):

        return state in self.invaderLocations


class DefendBorderANDLockOP1Problem(PositionSearchProblem):

    def __init__(self, gameState, agent):
        self._numExpanded = 0
        self._visitedLocations = set()
        self._visitHistory = []
        self.capsule = agent.getCapsules(gameState)
        self.startState = gameState.getAgentState(agent.index).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self.goal = []
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