from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from typing import NamedTuple
from collections import deque

from numpy import int8


class ThreeDifficultyLevelParameter(NamedTuple):
    wall: bool=True
    lava: bool=True
    door_key: bool=True
    max_step: int=1000
    num_channel: int=3
    history_length: int=1
    
TASKS = [
    ThreeDifficultyLevelParameter(wall=False, lava=False, door_key=False),
    ThreeDifficultyLevelParameter(wall=True, lava=False, door_key=False),
    ThreeDifficultyLevelParameter(wall=False, lava=True, door_key=False),
    ThreeDifficultyLevelParameter(wall=False, lava=False, door_key=True),
    ThreeDifficultyLevelParameter(wall=True, lava=True, door_key=False),
    ThreeDifficultyLevelParameter(wall=True, lava=False, door_key=True),
    ThreeDifficultyLevelParameter(wall=False, lava=True, door_key=True),
    ThreeDifficultyLevelParameter(wall=True, lava=True, door_key=True),
]


class ThreeDifficultyLevelEnvShort(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

    def __init__(self, params: ThreeDifficultyLevelParameter=ThreeDifficultyLevelParameter()):
        self.set_params(params)
        super().__init__(height=9, width=13, max_steps=params.max_step)
        self.observation_space = self.observation_space["image"]
        
        low = np.repeat(self.observation_space.low[:, :, :self.params.num_channel], self.params.history_length, axis=-1).reshape(-1)
        high = np.repeat(self.observation_space.high[:, :, :self.params.num_channel], self.params.history_length, axis=-1).reshape(-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    def reset_task(self, task: str):
        params = TASKS[task]
        self.set_params(params)

    def set_params(self, params: ThreeDifficultyLevelParameter):
        self.params = params

    def get_params(self):
        return self.params

    def _reward(self):
        return 10

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.params.door_key:
            # Doorkey challenge
            
            self.grid.vert_wall(5, 0)
            self.put_obj(Door('yellow', is_locked=False), 5, height - 2)
            
            self.place_obj(
                obj=Key('yellow'),
                top=(3, height - 2),
                size=(1, 1)
            )

        if self.params.wall:
            # Obstacle challange
            wall = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
            wall = np.array(wall)
            for x in range(wall.shape[1]):
                for y in range(wall.shape[0]):
                    if wall[y, x]:
                        self.put_obj(Wall(), x, y)

        if self.params.lava:
            # Lava challange
            lava = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
            lava = np.array(lava)
            for x in range(lava.shape[1]):
                for y in range(lava.shape[0]):
                    if lava[y, x]:
                        self.put_obj(Lava(), x, y)
                        
        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height // 2)

        self.mission = "reach the goal"
        self.agent_pos = (1, height // 2)
        self.agent_dir = 0
        self.last_x = None

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.ep_reward += reward
        obs = self._format_obs(obs)
        if done:
            info["eval_metrics"] = dict(reward=self.ep_reward)
        return obs, reward, done, info

    def reset(self):
        self.ep_reward = 0
        obs = super().reset()
        self.frames = deque([obs["image"][:, :, :self.params.num_channel]] * self.params.history_length, maxlen=self.params.history_length)
        self.past_dir = deque([self.agent_dir] * self.params.history_length, maxlen=self.params.history_length)
        obs = self._format_obs(obs)
        return obs

    def _format_obs(self, obs):
        obs = obs["image"][:, :, :self.params.num_channel]
        self.frames.append(obs)
        image = np.concatenate(self.frames, axis=-1).astype(np.float32)
        # self.past_dir.append(self.agent_dir)
        # return dict(
        #     image=image,
        #     direction=np.array(self.past_dir)
        # )
        return image.reshape(-1)
        


if __name__ == "__main__":
    import argparse
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    from gym_minigrid.window import Window

    def redraw(img):
        if not args.agent_view:
            img = env.render("rgb_array", tile_size=args.tile_size)

        window.show_img(img)

    def reset():
        if args.seed != -1:
            env.seed(args.seed)

        obs = env.reset()

        if hasattr(env, "mission"):
            print("Mission: %s" % env.mission)
            window.set_caption(env.mission)

        redraw(obs)

    def step(action):
        obs, reward, done, info = env.step(action)
        print(obs.shape)
        print("step=%s, reward=%.2f" % (env.step_count, reward))

        if done:
            print("done!")
            reset()
        else:
            redraw(obs)

    def key_handler(event):
        print("pressed", event.key)

        if event.key == "escape":
            window.close()
            return

        if event.key == "backspace":
            reset()
            return

        if event.key == "left":
            step(env.actions.left)
            return
        if event.key == "right":
            step(env.actions.right)
            return
        if event.key == "up":
            step(env.actions.forward)
            return

        # Spacebar
        if event.key == " ":
            step(env.actions.toggle)
            return
        if event.key == "pageup":
            step(env.actions.pickup)
            return
        if event.key == "pagedown":
            step(env.actions.drop)
            return

        if event.key == "enter":
            step(env.actions.done)
            return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    # env = ThreeDifficultyLevelEnvShort(ThreeDifficultyLevelParameter(False, False, False))
    from utils.make_env import make_env
    # env = make_env("minigrid_vector_discrete_full-v0", 0, params=ThreeDifficultyLevelParameter(False, False, False, True, True, True))()
    env = gym.make("MiniGrid-MultiRoom-N6-v0")
    obs = env.reset()
    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)

    reset()

    # Blocking event loop
    window.show(block=True)