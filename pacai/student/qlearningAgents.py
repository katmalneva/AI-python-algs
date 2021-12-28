from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
from pacai.util import probability
import random


class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: Basically I made a qValue dictionary that holds a [(state, action)],
    float pair. You can extract that value by calling getQValue. The update
    function updates the current qValue as the exploration/exploitation algorithm
    runs. The function getAction returns the probable action or the best
    action for that state. The function getPolicy returns the best action for
    the current state. The getValue function returns the max value for current state.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        self.qValues = counter.Counter()

    def getAction(self, state):
        if probability.flipCoin(self.getEpsilon()):
            return random.choice(self.getLegalActions(state))

        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        self.qValues[(state, action)] += \
            self.getAlpha() * ((reward + self.getDiscountRate() * self.getValue(nextState))
                               - self.qValues[(state, action)])

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        if (state, action) not in self.qValues:
            return 0.0

        return self.qValues[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        values = list()
        for action in self.getLegalActions(state):
            values.append(self.getQValue(state, action))

        if len(values) == 0:
            return 0.0

        return max(values)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = list()
        for action in self.getLegalActions(state):
            if self.getQValue(state, action) == self.getValue(state):
                actions.append(action)

        if len(actions) == 0:
            return None

        return random.choice(actions)


class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action


class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: I tried to do this but failed :/
    """

    def __init__(self, index, extractor='pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        self.weights = counter.Counter()
        self.stateWeights = counter.Counter()
        self.Qvalues = counter.Counter()

    def update(self, state, action, nextState, reward):
        correction = (reward + self.getDiscountRate()
                      * self.getValue(nextState)) - self.getQValue(state, action)

        self.weights[state] += self.getAlpha() * correction * (self.
            featExtractor.getFeatures(self, state, action)[state])
        self.stateWeights[action] += self.getAlpha() * correction * (self.
            featExtractor.getFeatures(self, state, action)[action])

    def getQValue(self, state, action):
        stateFeature = self.featExtractor.getFeatures(self, state, action)[state]
        actionFeature = self.featExtractor.getFeatures(self, state, action)[action]

        self.Qvalues[(state, action)] += \
            self.weights[state] * stateFeature + self.stateWeights[action] * actionFeature

        return self.Qvalues[(state, action)]

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)
        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            print()
