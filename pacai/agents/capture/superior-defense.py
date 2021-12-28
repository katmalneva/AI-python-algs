from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.bin.capture import AgentRules
from pacai.core.directions import Directions
from pacai.util import counter
from pacai.util.priorityQueue import PriorityQueue
from pacai.util.queue import Queue
from pacai.core.actions import Actions
from pacai.core.gamestate import AbstractGameState
from pacai.agents.capture.capture import CaptureAgent

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        # overwrite parent method to populates more useful fields
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

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        if myState.isScared():
            features['scared'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
            'scared': -40
        }

    def isGroove(self, actionList, action):
        if len(actionList) <= 2 and Actions.reverseDirection(action) in actionList:
            return True
        else:
            return False

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

"""        frontier = Queue()
        visited = set()
        result = None
        frontier.push((self.startingState(), [], 0))

        while not frontier.isEmpty():
            state, path, cost = frontier.pop()
            lastAction = path[-1]
            legalActions = list()
            for action in AgentRules.getLegalActions(problem, 0):
                legalActions.append(action)

            if len(legalActions) <= 2 and Actions.reverseDirection(lastAction) in legalActions:
                return path

            elif state not in visited:
                visited.add(state)
                for succState, succAction, succCost in problem.successorStates(state):
                    if succState not in visited:
                        newPath = path + [succAction]

                        frontier.push((succState, newPath, succCost))

        return result"""

""" def chooseAction(self, gameState):


        featureTotal = counter.Counter()
        for action in AgentRules.getLegalActions(gameState, 1):
            features = self.evaluate(gameState, action)
            featureTotal[action] = features
        print(featureTotal.argMax())

        return featureTotal.argMax()
"""