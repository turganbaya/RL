# main.py
import pygame
import numpy as np
import torch
import time
import argparse
from flappy_bird_game import FlappyBirdGame
from dqn_agent import DQNAgent

def train_agent(episodes=1000, render_every=50, save_every=100):
    """dqn agent"""
    game = FlappyBirdGame()
    agent = DQNAgent(state_size=4, action_size=2)
    
    scores = []
    losses = []
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        episode_losses = []
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            
            next_state, reward, done = game.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            if loss:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if episode % render_every == 0:
                game.render(q_value=agent.policy_net(
                    torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                ).max().item(), epsilon=agent.epsilon)
                pygame.display.flip()
                time.sleep(0.01)
        
        scores.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"episode {episode + 1}/{episodes}")
            print(f"  score: {game.score}, reward: {total_reward:.2f}, avg Score: {avg_score:.2f}")
            print(f"  epsilon: {agent.epsilon:.3f}, avg Loss: {avg_loss:.4f}")
            print("-" * 40)
        
        if (episode + 1) % save_every == 0:
            agent.save(f"flappy_bird_dqn_episode_{episode + 1}.pth")
            print(f"model saved at episode {episode + 1}")
    
    agent.save("flappy_bird_dqn_final.pth")
    game.close()
    
    return scores, losses

def play_with_ai(model_path=None, human_control=False):
    #trained ai or human
    game = FlappyBirdGame()
    
    if human_control:
        print("space to jump")
        agent = None
    else:
        agent = DQNAgent(state_size=4, action_size=2)
        if model_path:
            agent.load(model_path)
            agent.epsilon = 0.01 
        else:
            print("random agent")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if human_control and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    _, _, done = game.step(1)
                    if done:
                        game.reset()
        
        # ai control
        if not human_control:
            state = game.get_state()
            action = agent.select_action(state, training=False)
            _, _, done = game.step(action)
            
            if done:
                print(f"Game Over! Score: {game.score}")
                time.sleep(1)
                game.reset()
        
        q_value = None
        if agent and not game.game_over:
            state = game.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_value = agent.policy_net(state_tensor).max().item()
        
        game.render(q_value=q_value, epsilon=agent.epsilon if agent else None)
        # fps 60
        clock.tick(60)  
    
    game.close()

def evaluate_agent(model_path, num_games=100):
    game = FlappyBirdGame()
    agent = DQNAgent(state_size=4, action_size=2)
    agent.load(model_path)
    agent.epsilon = 0.01
    
    scores = []
    
    print(f"over {num_games} games...")
    
    for game_num in range(num_games):
        state = game.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, _, done = game.step(action)
        
        scores.append(game.score)
        
        if (game_num + 1) % 10 == 0:
            print(f"  Game {game_num + 1}/{num_games}: Score = {game.score}")
    
    game.close()
    
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    print(f"  avg score: {avg_score:.2f}")
    print(f"  best Score: {max_score}")
    print(f"  worst score: {min_score}")
    print(f"  games Played: {num_games}")
    
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flappy Bird AI Agent")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "play", "eval", "human"],
                       help="Mode: train, play, eval, or human")
    parser.add_argument("--model", type=str, default="flappy_bird_dqn_final.pth",
                       help="Path to model file")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent(episodes=args.episodes)
    elif args.mode == "play":
        play_with_ai(model_path=args.model)
    elif args.mode == "eval":
        evaluate_agent(model_path=args.model)
    elif args.mode == "human":
        play_with_ai(human_control=True)