import matplotlib
matplotlib.use('Agg')
import pytest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from warrl_env import GridWorldICM, visualize_paths_on_benchmark_maps

class DummyPolicy:
    def act(self, state_tensor):
        return np.random.randint(0, 4), None, None


def test_reset_shape():
    env = GridWorldICM(grid_size=6)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (5 * env.grid_size * env.grid_size,)


def test_reward_range():
    env = GridWorldICM(grid_size=6)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(1)
    assert env.reward_clip[0] <= reward <= env.reward_clip[1]


def test_visualization_runs(tmp_path):
    env = GridWorldICM(grid_size=6)
    env.reset(seed=0)

    from warrl_env import export_benchmark_maps
    export_benchmark_maps(env, num_maps=2, folder=tmp_path)

    class RandomPolicy:
        def act(self, state_tensor):
            return np.random.randint(0, 4), None, None

    visualize_paths_on_benchmark_maps(env, RandomPolicy(), map_folder=tmp_path, num_maps=2, grid_cols=1, save=False)
