import os
import argparse
import matplotlib.pyplot as plt
from snake_env import SnakeEnv
from ppo_agent import PPOAgent
from config import PPOConfig, FPS_PLAY


class TrainingVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.scores, self.avg_scores, self.lengths = [], [], []
        self.fig.suptitle('Snake PPO Training', fontsize=14, fontweight='bold')
    
    def update(self, episode, score, length, window=50):
        self.scores.append(score)
        self.lengths.append(length)
        self.avg_scores.append(sum(self.scores[-window:]) / min(len(self.scores), window))
        
        self.ax1.clear()
        self.ax1.plot(self.scores, alpha=0.3, color='blue', label='Score')
        self.ax1.plot(self.avg_scores, color='red', linewidth=2, label=f'Avg({window})')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Score')
        self.ax1.legend(loc='upper left')
        self.ax1.set_title(f'Episode {episode} | Best: {max(self.scores)}')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.plot(self.lengths, color='green', alpha=0.5)
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Snake Length')
        self.ax2.set_title(f'Max Length: {max(self.lengths)}')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def save(self, path='training_curve.png'):
        self.fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    
    def close(self):
        plt.ioff()
        plt.close()


def train(render=False):
    cfg = PPOConfig()
    env = SnakeEnv(render_mode='human' if render else None)
    agent = PPOAgent(env.state_dim, env.action_dim)
    vis = TrainingVisualizer()
    
    os.makedirs("models", exist_ok=True)
    best_score = 0
    best_avg = 0
    no_improve_count = 0
    
    for ep in range(cfg.NUM_EPISODES):
        state, done = env.reset(), False
        
        while not done:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.buffer.add(state, action, reward, value, log_prob, done)
            state = next_state
            
            if render and not env.render():
                vis.save()
                vis.close()
                return
            
            if len(agent.buffer.states) >= cfg.STEPS_PER_UPDATE:
                agent.update()
        
        if agent.buffer.states:
            agent.update()
        
        score, length = info['score'], env.snake.length
        vis.update(ep + 1, score, length)
        
        if score > best_score:
            best_score = score
            agent.save("models/best_model.pth")
        
        if len(vis.avg_scores) >= 50:
            current_avg = vis.avg_scores[-1]
            if current_avg > best_avg:
                best_avg = current_avg
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= 100 and current_avg < best_avg * 0.5:
                print(f"检测到崩溃，回滚到最佳模型...")
                agent.load("models/best_model.pth")
                no_improve_count = 0
        
        if (ep + 1) % cfg.LOG_INTERVAL == 0:
            print(f"Ep {ep+1}/{cfg.NUM_EPISODES} | Score: {score} | Avg: {vis.avg_scores[-1]:.1f} | Best: {best_score}")
        
        if (ep + 1) % cfg.SAVE_INTERVAL == 0:
            agent.save(f"models/model_{ep+1}.pth")
    
    agent.save("models/final_model.pth")
    vis.save()
    vis.close()
    env.close()
    print(f"Done! Best: {best_score}")


def play(model_path=None):
    env = SnakeEnv(render_mode='human')
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    path = model_path or "models/best_model.pth"
    if os.path.exists(path):
        agent.load(path)
        print(f"Loaded: {path}")
    
    while True:
        state, done = env.reset(), False
        while not done:
            action, _, _ = agent.get_action(state)
            state, _, done, info = env.step(action)
            if not env.render(fps=FPS_PLAY):
                env.close()
                return
        print(f"Score: {info['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=["train", "play"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args.render)
    else:
        play(args.model)
