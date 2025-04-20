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

frameStackSize = 4
env = gym.make("ALE/Boxing-v5",frameskip=1) #No frameskip as this is done in the preprocessing wrapper
noOfActions = env.action_space.n

EPSILON = 1
GAMMA = 0.99
C = 10000
EPISODES = 1500
BATCH_SIZE = 32
REPLAY_SIZE = 25000
LEARNING_RATE = 1e-4
SAVE_INTERVAL = 100 #Episodes

loss = 0 #For logging loss before we have recorded it

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

        self.optimiser = torch.optim.Adam(self.parameters(),lr=LEARNING_RATE)
    
    def forward(self,x):
        CNNOutput = self.convolutionalLayers(x)
        return self.fullyConnectedLayers(CNNOutput)




class DQNAgent():
    def __init__(self,env,exploration_rate,discount_factor,target_update_freq,training_episodes):

        self.policyNet = DeepQNetwork() #Init  policy and target net
        self.targetNet = DeepQNetwork()

        self.updateTargetNet() #Set target net to the same weights as policy net

        self.memory = deque(maxlen=REPLAY_SIZE) #Init replay memory

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

    def sampleMemory(self,BATCH_SIZE):
        sampleBatch = random.sample(self.memory,BATCH_SIZE)
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
        if len(self.memory) >= BATCH_SIZE:

            states, actions, rewards, next_states, dones = self.sampleMemory(BATCH_SIZE)
            states = torch.stack(states).to(device=device)
            actions = torch.tensor(actions).to(device=device)
            rewards = torch.tensor(rewards).to(device=device)
            next_states = torch.stack(next_states).to(device=device)
            dones = torch.tensor(dones).to(device=device)
            
            currentQs = self.policyNet(states).gather(1,actions.unsqueeze(1)) #Current values of sampled states from Q net. gather grabs actions taken. unsqueeze makes shape the same as states
            
            #print(currentQs.shape)
            #current values is 32 states and 18 actions
            #so .max(1)[0] gives the value of the maximum of each action, rather than e.g max(1)[1] which would be like argmax
            with torch.no_grad():

                doneMask = dones == False #create a mask representing whether a state is done (False) or not done (True)
                nextQMax = torch.zeros(BATCH_SIZE,device=device) #empty tensor for max values
                
                next_states = next_states[doneMask]

                nextQMax[doneMask] = self.targetNet(next_states).max(1)[0] #all max values, excluding done states
                targetQs = rewards + self.gamma * nextQMax # yj = rj + gamma * max actions

            #print(currentQs.shape,targetQs.shape) different shapes (32,1 / 32). may need to squeeze currentQs
            currentQs = currentQs.squeeze(1)
            
            #Finally update policy net values
            loss = lossFunction(currentQs,targetQs)

            self.policyNet.optimiser.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.policyNet.parameters(),100)

            self.policyNet.optimiser.step()
        
            return loss.item()


            
        
    def trainAgent(self,episodes):

        totalRewards = []
        episodeReward = 0

        for episode in range(0,episodes):

            state, info = self.env.reset()
            state = toTensor(state)
            
            T = count()

            print("Episode:",episode)            
            
            if self.epsilon > 0.1:
                self.epsilon *= 0.9955
                print("Epsilon:",self.epsilon)

            if episode > 0:
                file = open("log.txt","a")
                file.write(str((episode-1,episodeReward,loss,self.epsilon))+"\n")
                file.close()


            if episode % SAVE_INTERVAL == 0:
                #Save policy network
                filename = str("savedPolicyNet-" + str(episode) + ".pt")
                torch.save(self.policyNet.state_dict(), filename)

            episodeReward = 0

            for i in T:

                action = self.selectAction(state)
                
                next_state, reward, terminal, truncated, info = self.env.step(action)

                done = terminal or truncated
                episodeReward += reward

                if done:
                    break

                next_state = toTensor(next_state) #next state to tensor for consistency with state in memory

                self.saveMemory(state,action,reward,next_state,done) #Store transition s,a,r,s,d

                state = next_state

                
                # -- Update values of each sampled transition here --
                loss = self.runUpdate()
                
                if self.stepsComplete % self.C == 0:
                    self.updateTargetNet() #Update target net weights every C steps

                self.stepsComplete += 1




agent = DQNAgent(env,EPSILON,GAMMA,C,EPISODES)