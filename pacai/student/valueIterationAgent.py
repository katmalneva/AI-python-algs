from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter


class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0
        self.errorVal = 1 - discountRate

        for i in range(iters):
            self.newValues = counter.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    self.newValues[state] = 0
                    continue
                self.qvals = counter.Counter()

                for action in self.mdp.getPossibleActions(state):
                    qval = self.getQValue(state, action)
                    self.qvals[action] = qval
                    maxQval = self.qvals.argMax()
                    self.newValues[state] = self.qvals[maxQval]
            self.values = self.newValues

    def getQValue(self, state, action):
        """
        Should return Q(state,action).
        """
        val = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            vstar = self.getValue(nextState)
            val += prob * (self.mdp.getReward(state, action, nextState) + self.discountRate * vstar)

        return val

    def getPolicy(self, state):

        if self.mdp.isTerminal(state):
            return None

        policy = counter.Counter()
        for a in self.mdp.getPossibleActions(state):
            policy[a] = self.getQValue(state, a)

        action = policy.argMax()
        return action

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.getPolicy(state)
