# quick_start.py
import pygame
import random
import numpy as np

class SimpleFlappyBird:
    def __init__(self):
        self.WIDTH = 400
        self.HEIGHT = 600
        self.bird_y = self.HEIGHT // 2
        self.bird_velocity = 0
        self.gravity = 0.5
        self.jump_strength = -8
        self.pipes = []
        self.score = 0
        self.game_over = False
        
    def reset(self):
        self.bird_y = self.HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_over = False
        return self.get_state()
    
    def get_state(self):
        if not self.pipes:
            return [self.bird_y / self.HEIGHT, 0.5, 0.5, 0.5]
        
        next_pipe = self.pipes[0]
        return [
            self.bird_y / self.HEIGHT,
            next_pipe['gap_y'] / self.HEIGHT,
            (next_pipe['gap_y'] + 150) / self.HEIGHT,
            next_pipe['x'] / self.WIDTH
        ]
    
    def step(self, action):
        if action == 1:
            self.bird_velocity = self.jump_strength
        
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.WIDTH - 200:
            self.pipes.append({
                'x': self.WIDTH,
                'gap_y': random.randint(100, 400)
            })
        
        for pipe in self.pipes[:]:
            pipe['x'] -= 3
            if pipe['x'] < -80:
                self.pipes.remove(pipe)
                self.score += 1
        
        if self.bird_y > self.HEIGHT - 30 or self.bird_y < 0:
            self.game_over = True
            return self.get_state(), -10, True
        
        for pipe in self.pipes:
            if (pipe['x'] < 70 < pipe['x'] + 80 and 
                (self.bird_y < pipe['gap_y'] or self.bird_y > pipe['gap_y'] + 150)):
                self.game_over = True
                return self.get_state(), -10, True
        
        return self.get_state(), 1, False

class SimpleQLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = {}
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def get_q_key(self, state):
        discrete_state = tuple((np.array(state) * 10).astype(int))
        return discrete_state
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        key = self.get_q_key(state)
        if key not in self.q_table:
            self.q_table[key] = [0, 0]
        
        return np.argmax(self.q_table[key])
    
    def learn(self, state, action, reward, next_state, done):
        key = self.get_q_key(state)
        next_key = self.get_q_key(next_state)
        
        if key not in self.q_table:
            self.q_table[key] = [0, 0]
        if next_key not in self.q_table:
            self.q_table[next_key] = [0, 0]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_key])
        
        self.q_table[key][action] += self.learning_rate * (target - self.q_table[key][action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_simple_agent(episodes=500):
    game = SimpleFlappyBird()
    agent = SimpleQLearningAgent(state_size=4, action_size=2)
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        if (episode + 1) % 50 == 0:
            print(f"episode {episode + 1}/{episodes}, "
                  f"score: {game.score}, epsilon: {agent.epsilon:.3f}")
    
    return agent

if __name__ == "__main__":
    agent = train_simple_agent(episodes=500)
    
    game = SimpleFlappyBird()
    state = game.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        state, reward, done = game.step(action)
        print(f"bird Y: {game.bird_y:.1f}, score: {game.score}", end="\r")
    
    print(f"\nfinal Score: {game.score}")