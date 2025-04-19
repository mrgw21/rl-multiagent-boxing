import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torchvision.transforms as T
from collections import deque
import random
from itertools import count
import ale_py

from PrioritisedReplay_Latest import ReplayMemory

frameStackSize = 4
replaySize = 25000
learning_rate = 1e-4
batchSize = 32
env = gym.make("ALE/Boxing-v5",frameskip=1) #No frameskip as this is done in the preprocessing wrapper
noOfActions = env.action_space.n

device=torch.device("cuda") #Use GPUs

lossFunction = nn.HuberLoss() #Loss function for nn optimisation

def preProcess(env):
    env = AtariPreprocessing(env,frame_skip=4,screen_size=84,grayscale_obs=True) #Resize to 84x84 and greyscale
    env = FrameStackObservation(env,stack_size=frameStackSize) #Stack every four frames into one. can experiment with this variable
    return env #Wraps env so that states are pre-processed

def toTensor(state):
    return torch.tensor(state,dtype=torch.float32).to(device=device)


class DeepQNetwork(nn.Module):
    def __init__(self,inputShape=(4,84,84),noOfActions=18): #Default actions based on boxing
        super(DeepQNetwork,self).__init__()
        self.convolutionalLayers = nn.Sequential(
            # First conv layer: input channels = 4 (frame stack), output = 32
            nn.Conv2d(frameStackSize, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Second conv layer: 32 -> 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Third conv layer: 64 -> 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Fourth conv layer: 128 -> 128 filters
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fullyConnectedLayers = nn.Sequential(
            nn.Flatten(),
            # Fully connected layer to interpret features
            nn.Linear(128 * 5 * 5, 512), # that final output of the CNN is 128 feature maps of size 5x5
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout added - avoids overfitting
            # output logits for each action (18 actions)
            nn.Linear(512, noOfActions)
        )

        self.convolutionalLayers = nn.DataParallel(self.convolutionalLayers) #Make sure models use multi GPUs
        self.fullyConnectedLayers = nn.DataParallel(self.fullyConnectedLayers)

        self.convolutionalLayers.to(device=device) #Make sure models are on the correct device (cpu or gpu)
        self.fullyConnectedLayers.to(device=device)

        self.optimiser = torch.optim.Adam(self.parameters(),lr=learning_rate)
    
    def forward(self,x):
        CNNOutput = self.convolutionalLayers(x)
        return self.fullyConnectedLayers(CNNOutput)




class DQNAgent():
    def __init__(self,env,exploration_rate,discount_factor,target_update_freq,training_episodes):

        self.policyNet = DeepQNetwork() #Init  policy and target net
        self.targetNet = DeepQNetwork()

        self.updateTargetNet() #Set target net to the same weights as policy net

        #self.memory = deque(maxlen=replaySize) #Init replay memory
        self.memory = ReplayMemory(replaySize)

        self.epsilon = exploration_rate #Set hyperparams
        self.gamma = discount_factor
        self.C = target_update_freq

        self.env = preProcess(env) #Init environment, preprocessed by wrapper

        self.stepsComplete = 0

        self.trainAgent(training_episodes)

    def updateTargetNet(self):
        self.targetNet.load_state_dict(self.policyNet.state_dict())
    
    def saveMemory(self,state,action,reward,next_state,done):
        memory = (state,action,reward,next_state,done)
        self.memory.append(memory) #Add state to replay memory.

    def sampleMemory(self,batchSize):
        sampleBatch = random.sample(self.memory,batchSize)
        stateBatch, actionBatch, rewardBatch, nextStateBatch, doneBatch = zip(*sampleBatch)

        return stateBatch,actionBatch,rewardBatch,nextStateBatch,doneBatch
        #Currently just returning each individually. Should probably convert to tensors or np arrays

    
    def selectAction(self,state):

        if random.uniform(0,1) > self.epsilon:
            with torch.no_grad():
                state = state.unsqueeze(0) #Add extra dimension as only one state
                return self.policyNet(state).max(1).indices.view(1,1) #Return max-action

        else:
            return torch.tensor([[random.randint(0,noOfActions-1)]]).to(device=device)

    def testNNShape(self):
        state, info = self.env.reset()
        stateT = toTensor(state)
        
        output = self.policyNet(stateT) #Run through net

        if output.squeeze(0).shape == torch.Size([noOfActions]):
            print("NN outputs correct shape for 18 actions")
            return True
        else:
            print("Bad NN output!")
            return False
    
    def runUpdate(self):
        if len(self.memory.replay_buffer) >= batchSize:

            samples, sample_indices, importance_weights = self.memory.prioritised_sample()

            states, actions, rewards, next_states, dones = zip(*samples)

            states = torch.stack(states).to(device=device)
            actions = torch.tensor(actions).to(device=device)
            rewards = torch.tensor(rewards).to(device=device)
            next_states = torch.stack(next_states).to(device=device)
            dones = torch.tensor(dones).to(device=device)
            importance_weights = torch.tensor(importance_weights).to(device=device)
            
            currentQs = self.policyNet(states).gather(1,actions.unsqueeze(1)) #Current values of sampled states from Q net. gather grabs actions taken. unsqueeze makes shape the same as states
            
            #print(currentQs.shape)
            #current values is 32 states and 18 actions
            #so .max(1)[0] gives the value of the maximum of each action, rather than e.g max(1)[1] which would be like argmax
            with torch.no_grad():

                doneMask = dones == False #create a mask representing whether a state is done (False) or not done (True)
                nextQMax = torch.zeros(batchSize,device=device) #empty tensor for max values
                
                next_states = next_states[doneMask]

                nextQMax[doneMask] = self.targetNet(next_states).max(1)[0] #all max values, excluding done states
                targetQs = rewards + self.gamma * nextQMax # yj = rj + gamma * max actions
            
            #Finally update policy net values
            loss = lossFunction(currentQs.squeeze(1),targetQs)

            weightedLoss = (loss * importance_weights).mean()

            self.policyNet.optimiser.zero_grad()
            weightedLoss.backward()

            torch.nn.utils.clip_grad_value_(self.policyNet.parameters(),100)

            self.policyNet.optimiser.step()

            ## Calculate new TD errors

            with torch.no_grad():
                newQs = self.policyNet(states).gather(1,actions.unsqueeze(1)).squeeze(1) #Squeeze again to ensure shape of [32]
                TDErrors = targetQs - newQs
            
            TDErrors.squeeze(0)
            TDErrors = TDErrors.cpu() #It doesn't like GPU -> numpy
            

            priorities = [abs(TDErrors[i] + 1e-8) for i in range(len(TDErrors))] #Make sure new priorities are negative and not 0. List comp, thanks Haskell
            

            self.memory.update_priority_order(sample_indices,priorities)
            


            
        
    def trainAgent(self,episodes):

        totalRewards = []
        episodeReward : int
        epsilonArray = [self.epsilon]

        for i in range(0,episodes):

            state, info = self.env.reset()
            state = toTensor(state)
            
            T = count()

            if totalRewards!= []:
                print(totalRewards[i-1])
                file = open("totalRewards.txt","w")
                file.write(str(totalRewards))
                file.close()
            
            print("Episode", i)
            if i % 10 == 0:
                filename = str("savedPolicyNet" + str(i) + ".pt")
                torch.save(self.policyNet.state_dict(), filename)
            print("Saved")

            episodeReward = 0
            
            
            if self.epsilon > 0.1:
                self.epsilon *= 0.999
                epsilonArray.append(self.epsilon)
            print("Epsilon = ",self.epsilon)
            epsilonLog = open("epsilon.txt","w")
            epsilonLog.write(str(epsilonArray))
            epsilonLog.close()

            for i in T:

                #print("Step", i)

                action = self.selectAction(state)
                
                next_state, reward, terminal, truncated, info = self.env.step(action)

                done = terminal or truncated
                episodeReward += reward

                if done:
                    totalRewards.append(episodeReward)
                    break

                next_state = toTensor(next_state) #next state to tensor for consistency with state in memory

                #Get current state and next's Q for  initial TD Error

                with torch.no_grad():
                    currentQ = self.policyNet(state.unsqueeze(0))[0,action]
                    nextQ = self.policyNet(next_state.unsqueeze(0)).max(1).values
                    currentQ = currentQ.item()
                    nextQ = nextQ.item()

                    if done:
                        td_error = reward - currentQ
                    else:
                        td_error = reward + self.gamma * nextQ - currentQ

                experience = (state, action, reward, next_state, done)
                priority = abs(td_error)

                self.memory.store_experience(experience,priority,False)

                #self.saveMemory(state,action,reward,next_state,done) #Store transition s,a,r,s,d

                state = next_state

                
                # -- Update values of each sampled transition here --
                self.runUpdate()
                
                if self.stepsComplete % self.C == 0:
                    self.updateTargetNet() #Update target net weights every C steps

                self.stepsComplete += 1




agent = DQNAgent(env,1,0.99,10000,2000)