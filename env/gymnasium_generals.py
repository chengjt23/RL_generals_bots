from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generals.core.action import Action, compute_valid_move_mask
from generals.core.game import Game
from generals.core.grid import Grid, GridFactory
from generals.core.observation import Observation
from generals.core.replay import Replay
from generals.core.rewards import RewardFn, WinLoseRewardFn
from generals.gui import GUI
from generals.gui.properties import GuiMode


@dataclass
class AgentInfo:
    """Data structure to hold agent-specific information."""

    color: tuple[int, int, int]


class GymnasiumGenerals(gym.Env):
    """A Gymnasium environment for the Generals game."""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        agents: list[str],
        grid_factory: GridFactory | None = None,
        pad_observations_to: int = 24,
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory or GridFactory()
        self.reward_fn = reward_fn or WinLoseRewardFn()
        self.agents = agents
        self.truncation = truncation
        self.pad_observations_to = pad_observations_to

        self.agent_data = self._setup_agent_data()

        self.prior_observations: dict[str, Observation] | None = None
        grid = self.grid_factory.generate()
        self.game = Game(grid, self.agents)

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

    def _setup_agent_data(self) -> dict[str, dict[str, Any]]:
        colors = [(255, 107, 108), (0, 130, 255)]
        return {id: {"color": color} for id, color in zip(self.agents, colors)}

    def _create_observation_space(self) -> spaces.Space:
        dim = self.pad_observations_to
        return spaces.Box(low=0, high=2**31 - 1, shape=(2, 15, dim, dim), dtype=np.float32)

    def _create_action_space(self) -> spaces.Space:
        dim = self.pad_observations_to
        return spaces.MultiDiscrete([2, dim, dim, 4, 2])

    def _process_observations(self, observations: dict[str, Observation]) -> np.ndarray:
        processed_obs = []
        for agent in self.agents:
            observations[agent].pad_observation(pad_to=self.pad_observations_to)
            processed_obs.append(observations[agent].as_tensor())
        return np.stack(processed_obs)

    def _process_infos(
        self, observations: dict[str, Observation], game_infos: dict[str, Any], rewards: dict[str, float]
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            agent: {
                "army": np.array(game_infos[agent]["army"], dtype=np.int32),
                "land": np.array(game_infos[agent]["land"], dtype=np.int32),
                "done": np.array(game_infos[agent]["is_done"], dtype=bool),
                "winner": np.array(game_infos[agent]["is_winner"], dtype=bool),
                "masks": compute_valid_move_mask(observations[agent]),
                "reward": np.array(rewards[agent], dtype=np.float32),
            }
            for agent in self.agents
        }

    def _compute_rewards(self, actions: dict[str, Action], observations: dict[str, Observation]) -> dict[str, float]:
        assert self.prior_observations is not None, "Prior observations should always be legit."
        return {
            agent: self.reward_fn(
                prior_obs=self.prior_observations[agent],
                prior_action=actions[agent],
                obs=observations[agent],
            )
            for agent in self.agents
        }

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "grid" in options:
            grid = Grid(options["grid"])
        else:
            self.grid_factory.set_rng(rng=self.np_random)
            grid = self.grid_factory.generate()

        self.game = Game(grid, self.agents)

        if self.render_mode == "human":
            self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        raw_obs = {agent: self.game.agent_observation(agent) for agent in self.agents}
        observations = self._process_observations(raw_obs)
        self.prior_observations = raw_obs
        _infos = self.game.get_infos()
        _dummy_rewards = {agent: 0 for agent in self.agents}
        infos = self._process_infos(raw_obs, _infos, _dummy_rewards)

        return observations, infos

    def step(self, actions: list[Action] | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(actions, np.ndarray):
            # Convert numpy array to list of Action objects
            # Expected shape: (num_agents, 5) -> [to_pass, row, col, direction, split]
            action_objs = []
            for i in range(len(self.agents)):
                vals = actions[i]
                action_objs.append(Action(
                    to_pass=int(vals[0]),
                    row=int(vals[1]),
                    col=int(vals[2]),
                    direction=int(vals[3]),
                    to_split=bool(vals[4])
                ))
            actions = action_objs

        action_dict = {self.agents[i]: action for i, action in enumerate(actions)}

        observations, infos = self.game.step(action_dict)

        rewards = self._compute_rewards(action_dict, observations)
        processed_obs = self._process_observations(observations)
        processed_infos = self._process_infos(observations, infos, rewards)

        terminated = self.game.is_done()
        truncated = False if self.truncation is None else self.game.time >= self.truncation

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))
            if terminated or truncated:
                self.replay.store()

        self.prior_observations = {agent: observations[agent] for agent in self.agents}

        return processed_obs, 0.0, terminated, truncated, processed_infos

    def render(self) -> None:
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
