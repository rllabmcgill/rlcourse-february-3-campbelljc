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

import math
import mdp, util
import fileio, random
from collections import defaultdict

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, environment, discount = 0.9, iterations = 100, display=None):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.environment = environment
        
        self.action_vals = defaultdict(list)
        for state in self.mdp.getStates():
            self.action_vals[state] = [(0, action) for action in self.mdp.getPossibleActions(state)]
        
        self.num_state_updates = 0
        self.state_update_epoch = 10 #len(self.mdp.getStates())
        
        for i in range(self.iterations):
            print("Iteration", i)
            # copy the current values.
            new_values = util.Counter()
            
            # for each state
            for j, state in enumerate(self.mdp.getStates()):
                # compute max q-val over all actions
                max_qval = -100000
                self.action_vals[state] = []
                for action in self.mdp.getPossibleActions(state):
                    qval = self.computeQValueFromValues(state, action)
                    if qval > max_qval:
                        max_qval = qval
                        new_values[state] = max_qval # update value of current state to be the max
                    self.action_vals[state].append((qval, action))
                display.displayValues(self, state, "CURRENT VALUES", showActions=False)
                if i == 0 and j == 0: input()
                
                self.num_state_updates += 1
                if self.num_state_updates == self.state_update_epoch:
                    avg_return = self.compute_avg_return()
                    fileio.append(avg_return, "avg_returns_Value Iteration")
                    self.num_state_updates = 0
            
            self.values = new_values
        
    def compute_avg_return(self):
        # compute the avg. return from the current policy            
        num_returns = 100
        total_returns = 0
        
        decision = self.getAction
        for i in range(num_returns):
            returns = 0
            totalDiscount = 1.0
            self.environment.reset(None)
            if 'startEpisode' in dir(self): self.startEpisode()
            #message("BEGINNING EPISODE: "+str(episode)+"\n")
            timestep = 0
            MAX_TIMESTEPS = 20
            while True:
                if timestep >= MAX_TIMESTEPS:
                    break

                # DISPLAY CURRENT STATE
                state = self.environment.getCurrentState()
                #display(state)
                #pause()

                # END IF IN A TERMINAL STATE
                actions = self.environment.getPossibleActions(state)
                if len(actions) == 0:
                    #message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
                    break

                # GET ACTION (USUALLY FROM AGENT)
                action = decision(state)
                if action == None:
                    raise 'Error: Agent returned None action'

                # EXECUTE ACTION
                nextState, reward = self.environment.doAction(action)
                #message("Started in state: "+str(state)+
                #        "\nTook action: "+str(action)+
                #        "\nEnded in state: "+str(nextState)+
                #        "\nGot reward: "+str(reward)+"\n")
                # UPDATE LEARNER
                if 'observeTransition' in dir(self):
                    self.observeTransition(state, action, nextState, reward)

                returns += reward * totalDiscount
                totalDiscount *= self.discount
                timestep += 1

            if 'stopEpisode' in dir(self):
                self.stopEpisode()
                
            total_returns += returns
                        
        avg_returns = total_returns/num_returns
        return avg_returns
    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        
        qval = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += prob * (self.mdp.getReward(state, action, next_state) + (self.discount * self.values[next_state]))
    
        return qval

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None # as specified above
        
        action_vals = self.action_vals[state]
        
        # get max vals
        maxval = max(action_vals)[0]
        all_max = [action for qval, action in action_vals if qval == maxval]
        return random.choice(all_max) # random max

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class GSValueIterationAgent(ValueIterationAgent):
    def __init__(self, mdp, environment, discount = 0.9, iterations = 100, display=None):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.environment = environment

        self.action_vals = defaultdict(list)
        for state in self.mdp.getStates():
            self.action_vals[state] = [(0, action) for action in self.mdp.getPossibleActions(state)]
        
        self.num_state_updates = 0
        self.state_update_epoch = 10 #len(self.mdp.getStates())
        
        for i in range(self.iterations):            
            print("Iteration", i)
            # for each state
            for j, state in enumerate(self.mdp.getStates()):
                # compute max q-val over all actions
                max_qval = -100000
                self.action_vals[state] = []
                for action in self.mdp.getPossibleActions(state):
                    qval = self.computeQValueFromValues(state, action)
                    if qval > max_qval:
                        max_qval = qval
                        self.values[state] = max_qval # update value of current state to be the max
                    self.action_vals[state].append((qval, action))
                display.displayValues(self, state, "CURRENT VALUES", showActions=False)
                if i == 0 and j == 0: input()
                
                self.num_state_updates += 1
                if self.num_state_updates == self.state_update_epoch:
                    avg_return = self.compute_avg_return()
                    fileio.append(avg_return, "avg_returns_Gauss-Seidel")
                    self.num_state_updates = 0