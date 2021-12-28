import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.search import heuristic
from pacai.core.gamestate import AbstractGameState

from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        foodScore = 0
        ghostScore = 0
        scaredScore = 0
        stallScore = 0

        addedNumtoCord = -1

        # to prevent ghost from getting stuck in the same position
        if action == 'Stop':
            stallScore = 5

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()

        oldFood = currentGameState.getFood()
        for i in range(3):
            checkFoodx = newPosition[0]+addedNumtoCord
            checkFoody = newPosition[1]+addedNumtoCord

            if oldFood[checkFoodx][checkFoody]:
                foodScore = foodScore + 1
                # if current position is food increase foodScore +1 again
                if checkFoodx == newPosition[0]:
                    foodScore = foodScore + 1
            addedNumtoCord = addedNumtoCord + 1

        newGhostStates = successorGameState.getGhostStates()
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            if abs(newPosition[0]-ghostPos[0]) < 2 and abs(newPosition[1]-ghostPos[1]) < 2:
                ghostScore = 10

        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        for scared in newScaredTimes:
            if scared == 1:
                scaredScore = scaredScore + 3
                if ghostScore == 10:
                    # if there is a ghost nearby and it is scared add more to cancel out ghostScore
                    scaredScore = scaredScore + 6

        return successorGameState.getScore() - ghostScore + foodScore + scaredScore - stallScore

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, state):
        """The BaseAgent will receive an `pacai.core.gamestate.AbstractGameState`,
        and must return an action from `pacai.core.directions.Directions`."""

        action, move = self.minimax(state, 0, 0)
        return action

    def minimax(self, state, depth, index):
        action, move = self.maxVal(state, depth, index)
        return action, move

    def maxVal(self, state, depth, index):
        if depth == self.getTreeDepth() or state.isLose() or state.isWin():
            return Directions.STOP, self.getEvaluationFunction()(state)

        scores = []
        for a in state.getLegalActions():
            action, score = self.minVal(state.generateSuccessor(index, a), depth, index+1)
            scores.append(score)

        bestScore = max(scores)
        bestIndinces = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndinces)

        return state.getLegalActions(index)[chosenIndex], bestScore

    def minVal(self, state, depth, index):
        if depth == self.getTreeDepth() or state.isLose() or state.isWin():
            return Directions.STOP, self.getEvaluationFunction()(state)

        scores = []
        for a in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                action, score = self.maxVal(state.generateSuccessor(index, a), depth+1, 0)
                scores.append(score)

            else:
                action, score = self.minVal(state.generateSuccessor(index, a), depth, index+1)
                scores.append(score)

        bestScore = min(scores)
        bestIndinces = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndinces)

        return state.getLegalActions(index)[chosenIndex], bestScore






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, state):
        """The BaseAgent will receive an `pacai.core.gamestate.AbstractGameState`,
        and must return an action from `pacai.core.directions.Directions`."""

        action, move = self.alphaBeta(state, 0, 0)
        return action

    def alphaBeta(self, state, depth, index):
        action, move = self.maxVal(state, depth, index, -9999999, 9999999)
        return action, move

    def maxVal(self, state, depth, index, alpha, beta):
        if depth == self.getTreeDepth() or state.isLose() or state.isWin():
            return Directions.STOP, self.getEvaluationFunction()(state)

        scores = []
        for a in state.getLegalActions():
            action, score = self.minVal(state.generateSuccessor(index, a), depth, index + 1, alpha, beta)
            scores.append(score)
            bestScore = max(scores)
            alpha = max(alpha, bestScore)
            if bestScore >= beta:
                break

        bestIndinces = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndinces)

        return state.getLegalActions(index)[chosenIndex], bestScore

    def minVal(self, state, depth, index, alpha, beta):
        if depth == self.getTreeDepth() or state.isLose() or state.isWin():
            return Directions.STOP, self.getEvaluationFunction()(state)

        scores = []
        for a in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                action, score = self.maxVal(state.generateSuccessor(index, a), depth+1, 0, alpha, beta)
                scores.append(score)

            else:
                action, score = self.minVal(state.generateSuccessor(index, a), depth, index+1, alpha, beta)
                scores.append(score)

            bestScore = min(scores)
            beta = min(beta, bestScore)
            if bestScore <= alpha:
                break

        bestIndinces = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndinces)

        return state.getLegalActions(index)[chosenIndex], bestScore



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.
    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
