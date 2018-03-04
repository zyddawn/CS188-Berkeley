# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        cnt = 0
        while cnt < self.iterations:
            new_values = self.values.copy()   # in order not to change the state 
            cnt += 1
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    Q_values = [self.computeQValueFromValues(state, act) for act in actions]
                    new_values[state] = max(Q_values)
            self.values = new_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transit = self.mdp.getTransitionStatesAndProbs(state, action)
        avg_Qvalues = 0.0
        for s,p in transit:
            r = self.mdp.getReward(state, action, s)
            avg_Qvalues += p*(r + self.discount*self.getValue(s))
        return avg_Qvalues


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state):
            return None
        act_Qv = util.Counter()
        for act in actions:
            act_Qv[act] = self.computeQValueFromValues(state, act)
        return act_Qv.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            s = states[i % len(states)]
            if not self.mdp.isTerminal(s):
                act = self.computeActionFromValues(s)
                if act:
                    self.values[s] = self.computeQValueFromValues(s, act)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = util.Counter()
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        
        def getNewValue(s):
            act = self.computeActionFromValues(s)
            return self.computeQValueFromValues(s, act)
        
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for act in actions:
                nstates = self.mdp.getTransitionStatesAndProbs(s, act)
                for ns in nstates:
                    ns = ns[0]
                    if ns not in predecessors:
                        predecessors[ns] = set()
                    predecessors[ns].add(s)
        for s in states:
            if not self.mdp.isTerminal(s):
                new_value = getNewValue(s)
                diff = abs(new_value - self.values[s])
                pq.push(s, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            nv = getNewValue(s)
            self.values[s] = nv
            predcs = predecessors[s]
            for p in predcs:
                new_value = getNewValue(p)
                diff = abs(new_value - self.values[p])
                if diff > self.theta:
                    pq.update(p, -diff)









