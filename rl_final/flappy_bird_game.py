# flappy_bird_game.py
import pygame
import random
import numpy as np

class FlappyBirdGame:
    def __init__(self, width=400, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird - AI Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        self.gravity = 0.5
        self.jump_strength = -10
        self.pipe_speed = 3
        self.pipe_gap = 150
        self.pipe_frequency = 1500 
        self.last_pipe = pygame.time.get_ticks()
        
        self.reset()
    
    def reset(self):
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.score = 0
        self.pipes = []
        self.game_over = False
        self.last_pipe = pygame.time.get_ticks()
        return self.get_state()
    
    def get_state(self):
        if not self.pipes:
            return np.array([self.bird_y / self.height, 0, 0, 0])
        
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + 80 > 50:
                next_pipe = pipe
                break
        
        if next_pipe:
            return np.array([
                self.bird_y / self.height,
                next_pipe['top'] / self.height,
                next_pipe['bottom'] / self.height,
                (next_pipe['x'] / self.width)
            ])
        else:
            return np.array([self.bird_y / self.height, 0, 0, 0])
    
    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True
        
        if action == 1:
            self.bird_velocity = self.jump_strength
        
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe > self.pipe_frequency:
            pipe_height = random.randint(100, 400)
            self.pipes.append({
                'x': self.width,
                'top': pipe_height - self.pipe_gap,
                'bottom': pipe_height
            })
            self.last_pipe = current_time
            self.pipe_frequency = max(800, self.pipe_frequency - 10) 

        for pipe in self.pipes[:]:
            pipe['x'] -= self.pipe_speed
            if pipe['x'] < -80:
                self.pipes.remove(pipe)
                self.score += 1
 
        reward = 0.1  
        done = False
        
        if self.bird_y > self.height - 50 or self.bird_y < 0:
            reward = -10
            done = True
            self.game_over = True
       
        bird_rect = pygame.Rect(50, self.bird_y, 40, 30)
        for pipe in self.pipes:
            pipe_top_rect = pygame.Rect(pipe['x'], 0, 80, pipe['top'])
            pipe_bottom_rect = pygame.Rect(pipe['x'], pipe['bottom'], 80, self.height)
            
            if bird_rect.colliderect(pipe_top_rect) or bird_rect.colliderect(pipe_bottom_rect):
                reward = -10
                done = True
                self.game_over = True
                break
        
        for pipe in self.pipes:
            if pipe['x'] + 80 == 50:
                reward = 5
                break
        
        return self.get_state(), reward, done
    
    def render(self, q_value=None, epsilon=None):
        self.screen.fill((135, 206, 235)) 
        
        pygame.draw.rect(self.screen, (222, 184, 135), (0, self.height - 50, self.width, 50))
        pygame.draw.rect(self.screen, (139, 69, 19), (0, self.height - 50, self.width, 10))
        
        bird_color = (255, 255, 0) if not self.game_over else (255, 0, 0)
        pygame.draw.circle(self.screen, bird_color, (70, int(self.bird_y)), 15)
        pygame.draw.circle(self.screen, (0, 0, 0), (80, int(self.bird_y - 5)), 5)  
        
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 128, 0), (pipe['x'], 0, 80, pipe['top']))
            pygame.draw.rect(self.screen, (0, 100, 0), (pipe['x'], pipe['top'] - 20, 80, 20))
            
            pygame.draw.rect(self.screen, (0, 128, 0), (pipe['x'], pipe['bottom'], 80, self.height))
            pygame.draw.rect(self.screen, (0, 100, 0), (pipe['x'], pipe['bottom'], 80, 20))
        
        score_text = self.font.render(f'score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        if q_value is not None:
            q_text = self.font.render(f'q: {q_value:.3f}', True, (255, 255, 255))
            self.screen.blit(q_text, (10, 40))
        
        if epsilon is not None:
            eps_text = self.font.render(f'eps: {epsilon:.3f}', True, (255, 255, 255))
            self.screen.blit(eps_text, (10, 70))
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()