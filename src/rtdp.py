from game import *
from learningAgents import ValueEstimationAgent
import fileio

import random,util,math
from collections import defaultdict

class RTDPLearningAgent(ValueEstimationAgent):
    def __init__(self, mdp, env, discount = 0.9, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.environment = env
        
        # determine subset of possible start states
        self.start_states = set()
        for x in range(env.gridWorld.grid.width):
            for y in range(env.gridWorld.grid.height):
                if env.gridWorld.grid[x][y] in [' ', 'S']:
                    self.start_states.add((x, y))
        assert len(self.start_states) > 0
        self.start_states = list(self.start_states)
        
        #env.setAgent(self)
        
        self.states_to_backup = set() # (state, action, reward, next_state) tuples
        self.action_vals = defaultdict(list)
        for state in self.mdp.getStates(): # init
            for action in self.mdp.getPossibleActions(state):
                self.action_vals[state].append((0, action))
        
        self.num_state_updates = 0
        self.state_update_epoch = 10 #len(self.mdp.getStates())        
    
    def getStartState(self):
        return random.choice(self.start_states)
    
    def update(self, state, action, nextState, reward):
        self.states_to_backup.add((state, action, nextState, reward))
        
        print(len(self.mdp.getStates()))
        
        changed_states = []
        for state, action_, next_state_, reward in self.states_to_backup:
            # compute max q-val over all actions
            max_qval = -100000
            self.action_vals[state] = []
            for action in self.mdp.getPossibleActions(state):
                qval = self.computeQValueFromValues(state, action, reward=reward if action == action_ else None)
                if qval > max_qval:
                    max_qval = qval
                    #self.best_actions[state] = action # store best action for policy
                    if self.values[state] != max_qval:
                        changed_states.append((state, action_, next_state_, reward))
                    self.values[state] = max_qval # update value of current state to be the max
                self.action_vals[state].append((qval, action))
            
            self.num_state_updates += 1
        
            if self.num_state_updates == self.state_update_epoch:
                avg_return = self.compute_avg_return()
                fileio.append(avg_return, "avg_returns_RTDP")
                self.num_state_updates = 0
        
        self.states_to_backup = set()
        
    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)
    
    def observeTransition(self, state,action,nextState,deltaReward):
        self.update(state,action,nextState,deltaReward)
    
    def compute_avg_return(self):
        # compute the avg. return from the current policy            
        num_returns = 100
        total_returns = 0
        
        decision = self.getAction
        for i in range(num_returns):
            returns = 0
            totalDiscount = 1.0
            self.environment.reset()
            if 'startEpisode' in dir(self): self.startEpisode()
            #message("BEGINNING EPISODE: "+str(episode)+"\n")
            while True:

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

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
    
    def computeQValueFromValues(self, state, action, reward=None):
        qval = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            if reward is None:
                reward = self.mdp.getReward(state, action, next_state)
            qval += prob * (reward + (self.discount * self.values[next_state]))
    
        return qval

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None # as specified above
        
        action_vals = self.action_vals[state]
        
        # get max vals
        maxval = max(action_vals)[0]
        all_max = [action for qval, action in action_vals if qval == maxval]
        return random.choice(all_max) # random max

    def compute_avg_return(self):
        # compute the avg. return from the current policy            
        num_returns = 100
        total_returns = 0
        
        decision = self.getAction
        for i in range(num_returns):
            returns = 0
            totalDiscount = 1.0
            self.environment.reset(None) # start at the start state.
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
                
                # UPDATE LEARNER
                #if 'observeTransition' in dir(self):
                #    self.observeTransition(state, action, nextState, reward)

                returns += reward * totalDiscount
                totalDiscount *= self.discount
                timestep += 1

            if 'stopEpisode' in dir(self):
                self.stopEpisode()
                
            total_returns += returns
                        
        avg_returns = total_returns/num_returns
        return avg_returns