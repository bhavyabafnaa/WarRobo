import numpy as np
import matplotlib.patches as patches
import gym
import torch
import os
from matplotlib.collections import LineCollection

class GridWorldICM:
    def __init__(self, grid_size=6, dynamic_risk=False, reward_clip=(-10, 100), max_steps=100, seed=None):
        self.grid_size = grid_size
        self.dynamic_risk = dynamic_risk
        self.reward_clip = reward_clip
        self.max_steps = max_steps
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return seed

    def reset(self, seed=None, load_map_path=None, add_noise=False):
        if seed is not None:
            self.seed(seed)

        if load_map_path:
            self.load_map(load_map_path)
        else:
            self.cost_map = np.ones((self.grid_size, self.grid_size)) * 0.3
            self.risk_map = np.zeros((self.grid_size, self.grid_size))
            self.terrain_map = np.full((self.grid_size, self.grid_size), 'normal')
            self.mine_map = self.np_random.random((self.grid_size, self.grid_size)) < 0.05
            self.terrain_map[self.np_random.random((self.grid_size, self.grid_size)) < 0.15] = 'mud'
            self.terrain_map[self.np_random.random((self.grid_size, self.grid_size)) < 0.15] = 'water'
            self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
            self.enemy_positions = [[self.grid_size // 2, self.grid_size // 2]]

        if add_noise:
          noise = np.random.normal(0, 0.05, self.risk_map.shape)
          self.risk_map = np.clip(self.risk_map + noise, 0.0, 1.0)

          noise = np.random.normal(0, 0.05, self.cost_map.shape)
          self.cost_map = np.clip(self.cost_map + noise, 0.0, 1.0)

        self.agent_pos = [0, 0]
        self.prev_distance_to_goal = self._distance_to_goal(self.agent_pos)
        self.steps = 0

        return self._get_obs(), {}

    def load_map(self, filepath="map_data.npz"):
        data = np.load(filepath, allow_pickle=True)
        self.cost_map = data["cost_map"]
        self.risk_map = data["risk_map"]
        self.terrain_map = data["terrain_map"]
        self.mine_map = data["mine_map"]
        self.goal_pos = list(data["goal_pos"])
        self.enemy_positions = data["enemy_positions"].tolist()
        self.grid_size = self.cost_map.shape[0]

    def save_map(self, filepath="map_data.npz"):
        np.savez_compressed(filepath,
                            cost_map=self.cost_map,
                            risk_map=self.risk_map,
                            terrain_map=self.terrain_map,
                            mine_map=self.mine_map,
                            goal_pos=self.goal_pos,
                            enemy_positions=np.array(self.enemy_positions))

    def _distance_to_goal(self, pos):
        return np.linalg.norm(np.array(pos) - np.array(self.goal_pos))

    def step(self, action, terrain_decay=1.0):
        x, y = self.agent_pos
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.grid_size - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.grid_size - 1: y += 1
        self.agent_pos = [x, y]

        self.steps += 1
        if self.dynamic_risk and self.np_random.rand() < 0.1:
            i, j = self.np_random.randint(self.grid_size, size=2)
            self.risk_map[i][j] = min(1.0, self.risk_map[i][j] + 0.5)

        for enemy in self.enemy_positions:
            dx, dy = self.np_random.choice([-1, 0, 1], size=2)
            enemy[0] = np.clip(enemy[0] + dx, 0, self.grid_size - 1)
            enemy[1] = np.clip(enemy[1] + dy, 0, self.grid_size - 1)

        cost = np.clip(self.cost_map[x][y], 0, 1)
        risk = np.clip(self.risk_map[x][y], 0, 1)

        reward = -cost - risk
        terrain = self.terrain_map[x][y]
        if terrain == 'mud':
            reward -= 0.2 * terrain_decay
        elif terrain == 'water':
            reward -= 0.1 * terrain_decay

        if self.mine_map[x][y]: reward -= 2

        for ex, ey in self.enemy_positions:
            if [x, y] == [ex, ey]:
                reward -= 3
                return self._get_obs(), self._clip_reward(reward), True, False, {}

        dist = self._distance_to_goal([x, y])
        if dist < self.prev_distance_to_goal:
            reward += 0.2
        self.prev_distance_to_goal = dist

        done = [x, y] == self.goal_pos or self.steps >= self.max_steps
        if [x, y] == self.goal_pos:
            reward += 50

        return self._get_obs(), self._clip_reward(reward), done, False, {}

    def _clip_reward(self, r):
        return np.clip(r, self.reward_clip[0], self.reward_clip[1])

    def _get_obs(self):
        obs = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs[1] = self.cost_map
        obs[2] = self.risk_map
        obs[3, self.goal_pos[0], self.goal_pos[1]] = 1.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.terrain_map[i][j] == 'mud': obs[4, i, j] = 0.5
                elif self.terrain_map[i][j] == 'water': obs[4, i, j] = 0.8
                if self.mine_map[i][j]: obs[4, i, j] = 1.0
        return obs.flatten()

    def render(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.axis('off')

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = 'white'
                if self.terrain_map[i][j] == 'mud': color = '#C2B280'
                elif self.terrain_map[i][j] == 'water': color = '#99ccff'
                if self.mine_map[i][j]: color = '#ff9999'
                if self.risk_map[i][j] > 0: color = '#ffcccb'
                ax.add_patch(patches.Rectangle((j, self.grid_size - 1 - i), 1, 1, facecolor=color, edgecolor='gray'))

        gx, gy = self.goal_pos
        ax.add_patch(patches.Rectangle((gy, self.grid_size - 1 - gx), 1, 1, facecolor='lime', edgecolor='black'))
        for ex, ey in self.enemy_positions:
            ax.add_patch(patches.Circle((ey + 0.5, self.grid_size - 1 - ex + 0.5), 0.3, color='red'))
        ax.add_patch(patches.Circle((self.agent_pos[1] + 0.5, self.grid_size - 1 - self.agent_pos[0] + 0.5), 0.3, color='black'))

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]
        plt.close(fig)
        return img



import matplotlib.pyplot as plt
import numpy as np
import os


from matplotlib.collections import LineCollection

def visualize_paths_on_benchmark_maps(env, policy, map_folder="maps/", num_maps=4, grid_cols=2, save=False):
    grid_rows = int(np.ceil(num_maps / grid_cols))
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 4), dpi=300)
    axs = axs.flatten()

    for i in range(num_maps):
        map_path = os.path.join(map_folder, f"map_{i:02d}.npz")
        obs, _ = env.reset(load_map_path=map_path)
        path = [tuple(env.agent_pos)]
        rewards = []

        done = False
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = policy.act(state_tensor)
            obs, reward, done, _, _ = env.step(action)
            path.append(tuple(env.agent_pos))
            rewards.append(reward)

        # Create a line with colors based on rewards
        segments = []
        for j in range(len(path) - 1):
            segments.append([ (path[j][1], path[j][0]), (path[j+1][1], path[j+1][0]) ])
        rewards = np.array(rewards)
        norm = plt.Normalize(rewards.min(), rewards.max())
        lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=2)
        lc.set_array(rewards)

        ax = axs[i]
        ax.imshow(env.cost_map, cmap='Blues', alpha=0.3, origin='lower')
        ax.imshow(env.risk_map, cmap='Reds', alpha=0.3, origin='lower')
        ax.add_collection(lc)
        ax.plot(env.goal_pos[1], env.goal_pos[0], marker='*', color='lime', markersize=10)
        ax.set_title(f"Map {i}")
        ax.axis("off")

        if save:
            plt.savefig(f"benchmark_visuals/map_{i:02d}.png")

    for j in range(num_maps, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Reward-Weighted Agent Paths", fontsize=16, y=1.02)
    plt.show()



def export_benchmark_maps(env, num_maps=10, folder="maps/"):
    os.makedirs(folder, exist_ok=True)  # ✅ Ensure directory exists
    for i in range(num_maps):
        env.reset(seed=i)
        env.save_map(f"{folder}/map_{i:02d}.npz")

def evaluate_on_benchmarks(env, policy, map_folder="maps/", num_maps=10):
    rewards = []
    for i in range(num_maps):
        map_path = f"{map_folder}/map_{i:02d}.npz"
        obs, _ = env.reset(load_map_path=map_path)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = policy.act(state_tensor)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

def plot_model_performance(models, model_names, env, map_folder="maps/", num_maps=10):
    avg_rewards = []
    std_devs = []

    for policy in models:
        mean_r, std_r = evaluate_on_benchmarks(env, policy, map_folder, num_maps)
        avg_rewards.append(mean_r)
        std_devs.append(std_r)

    x = np.arange(len(model_names))
    plt.figure(figsize=(8, 5))
    plt.bar(x, avg_rewards, yerr=std_devs, capsize=5, color=['gray', 'skyblue', 'green'])
    plt.xticks(x, model_names)
    plt.ylabel("Average Reward")
    plt.title("Model Performance on Benchmark Maps")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()



