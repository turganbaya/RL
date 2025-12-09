чтобы запустить обучение агента:
python main.py --mode train --episodes 1000

чтобы поиграть обученным агентом:
python main.py --mode play --model flappy_bird_dqn_final.pth

чтобы поиграйте самому:
python main.py --mode human

производительность агента:
python main.py --mode eval --model flappy_bird_dqn_final.pth
