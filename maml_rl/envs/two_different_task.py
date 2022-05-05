import gym
import numpy as np
from enum import IntEnum
from gym_minigrid.roomgrid import RoomGrid, Ball, Door, Lava
from IPython.core.debugger import set_trace


class Tasks(IntEnum):
    unlock = 0
    upperTunel = 1


class TwoDifferentTask(RoomGrid):
    def __init__(self, seed=None):
        room_size = 8
        #set_trace()
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=1000, #8*room_size**3*9
            seed=seed
        )
        print("init")
        self.current_task = Tasks.unlock
        self.door_closed = False
        self.room_size = room_size
        self.target_type = 'ball'

        self.observation_space = self.observation_space["image"]
        #chanel_index = 3
        #history_length = 1
        #low = np.repeat(self.observation_space.low[:, :, :channel_index], history_length, axis=-1).reshape(-1)
        #high = np.repeat(self.observation_space.high[:, :, :channel_index], history_length, axis=-1).reshape(-1)
        self.observation_space = gym.spaces.Box(low=self.observation_space.low.reshape(-1), high=self.observation_space.high.reshape(-1), dtype=np.float32)
        
        #print("init action space", self.action_space)
        #low = np.repeat(self.observation_space.low[:, :, :self.params.num_channel], self.params.history_length, axis=-1).reshape(-1)
        #high = np.repeat(self.observation_space.high[:, :, :self.params.num_channel], self.params.history_length, axis=-1).reshape(-1)
        #self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        #print("my obs shape", self.observation_space.shape)

    def _gen_grid_unlock(self, width, height):
        #self.num_cols = 2
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door1, _ = self.add_door(0, 0, 0, locked=True, color='red')
        # Add a key to unlock the door
        key, _ = self.add_object(0, 0, 'key', "red")
        self.place_agent(0, 0)
        self.door1 = door1

    def _gen_obs_for_each_room(self, room_col_index, room_row_index):
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+2, 2)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+4, 2)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+6, 1)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+1, 3)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+3, 4)

    def _gen_grid_upper_tunel(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        #door1, _ = self.add_door(0, 0, 0, locked=False, color='red')
        #door2, _ = self.add_door(1, 0, 0, locked=True, color='red')
        # Add a key to unlock the door
        #key, _=self.add_object(1, 0, 'key', "red")
        #ball, _=self.add_object(1, 0, 'ball', "red")
        #print(ball.init_pos)
        #ball.cur_pos=[1, 1]#self.room_size+4, 1]
        self.grid.set(self.room_size+3, 1, Ball())

        key, _ = self.add_object(0, 0, 'key', "red")
        #self.grid.set(2*(self.room_size-1)+self.room_size//2,self.room_size//2, Goal())

        #goal_key, _=self.add_object(1, 0, 'key', 'green')
        #box, _=self.add_object(1, 0, 'box', 'blue')
        #box.contains = goal_key

        # make this two rooms connected via a gap
        self.grid.set(self.room_size-1, self.room_size//4, None)
        #self.grid.set(self.room_size-1, self.room_size//4-1, None)
        self.grid.set(self.room_size-1, 3*self.room_size //
                      4-1, Door("red", is_locked=True))
        self.grid.set(self.room_size*2-2, 3*self.room_size//4-1, None)
        #self.grid.set(2*(self.room_size-1),self.room_size//2, None)
        self._gen_obs_for_each_room(1, 0)
        self._gen_obs_for_each_room(0, 0)

        self.place_agent(0, 0)
        #self.agent_pos=[1, 4]
        #print("agent pos",self.agent_pos)

        #self.door1 = door1
        #self.door1=door1
        #self.door2=door2
        #self.box=box
        #self.goal_key=goal_key

    def _gen_grid(self, width, height):

        if self.current_task == Tasks.unlock:
            self._gen_grid_unlock(width, height)
        elif self.current_task == Tasks.upperTunel:
            self._gen_grid_upper_tunel(width, height)

    def reset_task(self, task: int):
        #print("reset task", task)
        self.current_task = task
        #if task == Tasks.unlock:
        #    #self._gen_grid = self._gen_grid_unlock
        #    #self.step = self.step_unlock
        #elif task == Tasks.upperTunel:
        #    #self._gen_grid = self._gen_grid_upper_tunel
        #    #self.step = self.step_upper_tunel

    def step_unlock(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.toggle:
            if self.door1.is_open:  # and self.door2.is_open:
                reward = self._reward()
                done = True

        #print("task1",self.door2.is_open)

        return obs, reward, done, info

    def step_upper_tunel(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying:
            if (self.carrying.type == self.target_type):
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True
    def step(self, action):
        if self.current_task == Tasks.unlock:
            return self.step_unlock(action)
        elif self.current_task == Tasks.upperTunel:
            return self.step_upper_tunel(action)
        else:
            print("Task not recognized")
            exit(-1)
    def reset(self):
        #print("reset action space", self.action_space)
        print("step")
        image = super().reset()["image"]
        obs = image.reshape(-1)
        return obs

