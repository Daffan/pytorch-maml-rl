from gym_minigrid.minigrid import Ball
from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register
from pdb import set_trace

class SmallUnlock(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 8 
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**3*9,
            seed=seed
        )
        self.door_closed=False
        self.room_size=room_size

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door1, _ = self.add_door(0, 0, 0, locked=True, color='red')
        #door2, _ = self.add_door(1, 0, 0, locked=True, color='red')
        # Add a key to unlock the door
        key, _=self.add_object(0, 0, 'key', "red")
        #ball, _=self.add_object(1, 0, 'ball', "red")

        #self.grid.set(2*(self.room_size-1)+self.room_size//2,self.room_size//2, Goal())

        #goal_key, _=self.add_object(1, 0, 'key', 'green') 
        #box, _=self.add_object(1, 0, 'box', 'blue')
        #box.contains = goal_key

        # make this two rooms connected via a gap 
        #self.grid.set(self.room_size-1, self.room_size//2, None)
        #self.grid.set(2*(self.room_size-1),self.room_size//2, None)

        self.place_agent(0, 0)
        #self.agent_pos=[1, 1]
        #print("agent pos",self.agent_pos)

        #self.door1 = door1
        #self.door1=door1
        self.door1=door1
        #self.box=box
        #self.goal_key=goal_key

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        if action == self.actions.toggle:
            if self.door1.is_open: # and self.door2.is_open:
                reward = self._reward()
                done = True

        #print("task1",self.door2.is_open)

        return obs, reward, done, info



class UpperTunel(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 8 
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=8*room_size**3*9,
            seed=seed
        )
        self.door_closed=False
        self.room_size=room_size
        self.target_type = 'ball' 

    def _gen_obs_for_each_room(self, room_col_index, room_row_index):
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+2,2)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+4,2)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+6,1)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+1,3)
        #self.put_obj(Lava(), (self.room_size-1)*room_col_index+1,1)
        self.put_obj(Lava(), (self.room_size-1)*room_col_index+3,4)

    def _gen_grid(self, width, height):
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

        key, _=self.add_object(0, 0, 'key', "red")
        #self.grid.set(2*(self.room_size-1)+self.room_size//2,self.room_size//2, Goal())

        #goal_key, _=self.add_object(1, 0, 'key', 'green') 
        #box, _=self.add_object(1, 0, 'box', 'blue')
        #box.contains = goal_key

        # make this two rooms connected via a gap 
        self.grid.set(self.room_size-1, self.room_size//4, None)
        #self.grid.set(self.room_size-1, self.room_size//4-1, None)
        self.grid.set(self.room_size-1, 3*self.room_size//4-1, Door("red", is_locked=True))
        self.grid.set(self.room_size*2-2, 3*self.room_size//4-1, None)
        #self.grid.set(2*(self.room_size-1),self.room_size//2, None)
        self._gen_obs_for_each_room(1,0)
        self._gen_obs_for_each_room(0,0)

        self.place_agent(0, 0)
        #self.agent_pos=[1, 4]
        #print("agent pos",self.agent_pos)
    

        #self.door1 = door1
        #self.door1=door1
        #self.door2=door2
        #self.box=box
        #self.goal_key=goal_key

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying:
            if (self.carrying.type == self.target_type):
                reward = self._reward() 
                done = True
            else: 
                reward = 0 
                done = True

        #if action == self.actions.toggle:
        #    if self.door2.is_open: # and self.door2.is_open:
        #        reward = self._reward()
        #        done = True

        #print("task1",self.door2.is_open)

        return obs, reward, done, info

class LowerTunel(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 8 
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=8*room_size**3*3,
            seed=seed
        )
        self.door_closed=False
        self.room_size=room_size
        self.target_type = 'ball' 

    def _gen_obs_for_each_room(self, room_col_index, room_row_index):
        self.put_obj(Lava(), self.room_size*room_col_index+1,2)
        self.put_obj(Lava(), self.room_size*room_col_index+3,2)
        self.put_obj(Lava(), self.room_size*room_col_index+5,1)
        self.put_obj(Lava(), self.room_size*room_col_index,3)
        self.put_obj(Lava(), self.room_size*room_col_index-1,1)
        self.put_obj(Lava(), self.room_size*room_col_index+2,4)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        #door1, _ = self.add_door(0, 0, 0, locked=False, color='red')
        #door2, _ = self.add_door(1, 0, 0, locked=True, color='red')
        # Add a key to unlock the door
        #key, _=self.add_object(1, 0, 'key', "red")
        #ball, _=self.add_object(1, 0, 'ball', "red")
        #print(ball.init_pos)
        #ball.cur_pos=[1, 1]#self.room_size+4, 1]
        self.grid.set(self.room_size*2+3, 5, Ball())

        #self.grid.set(2*(self.room_size-1)+self.room_size//2,self.room_size//2, Goal())

        #goal_key, _=self.add_object(1, 0, 'key', 'green') 
        #box, _=self.add_object(1, 0, 'box', 'blue')
        #box.contains = goal_key

        # make this two rooms connected via a gap 
        self.grid.set(self.room_size-1, self.room_size//4, None)
        self.grid.set(self.room_size-1, self.room_size//4-1, None)
        self.grid.set(self.room_size-1, 3*self.room_size//4-1, None)
        self.grid.set(self.room_size*2-2, 3*self.room_size//4-1, None)
        #self.grid.set(2*(self.room_size-1),self.room_size//2, None)
        self._gen_obs_for_each_room(1,0)

        self.place_agent(0, 0)
        self.agent_pos=[1, 1]
        #print("agent pos",self.agent_pos)
    

        #self.door1 = door1
        #self.door1=door1
        #self.door2=door2
        #self.box=box
        #self.goal_key=goal_key

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.carrying:
            if (self.carrying.type == self.target_type):
                reward = self._reward() 
                done = True
            else: 
                reward = 0 
                done = True

        #if action == self.actions.toggle:
        #    if self.door2.is_open: # and self.door2.is_open:
        #        reward = self._reward()
        #        done = True

        #print("task1",self.door2.is_open)

        return obs, reward, done, info


class MyUnlock1(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 8 
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=8*room_size**3*3,
            seed=seed
        )
        self.door_closed=False
        self.room_size=room_size

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        #door1, _ = self.add_door(0, 0, 0, locked=False, color='red')
        door2, _ = self.add_door(1, 0, 0, locked=True, color='red')
        # Add a key to unlock the door
        key, _=self.add_object(1, 0, 'key', "red")
        ball, _=self.add_object(1, 0, 'ball', "red")

        #self.grid.set(2*(self.room_size-1)+self.room_size//2,self.room_size//2, Goal())

        #goal_key, _=self.add_object(1, 0, 'key', 'green') 
        #box, _=self.add_object(1, 0, 'box', 'blue')
        #box.contains = goal_key

        # make this two rooms connected via a gap 
        self.grid.set(self.room_size-1, self.room_size//2, None)
        #self.grid.set(2*(self.room_size-1),self.room_size//2, None)

        self.place_agent(0, 0)
        self.agent_pos=[1, 1]
        print("agent pos",self.agent_pos)

        #self.door1 = door1
        #self.door1=door1
        self.door2=door2
        #self.box=box
        #self.goal_key=goal_key

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        if action == self.actions.toggle:
            if self.door2.is_open: # and self.door2.is_open:
                reward = self._reward()
                done = True

        #print("task1",self.door2.is_open)

        return obs, reward, done, info

class MyUnlock2(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 8 
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=8*room_size**3*3,
            seed=seed
        )
        self.door_closed=False
        self.room_size=room_size
        self.target_type = 'ball' 
        self.target_color = 'red' 


    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        #door1, _ = self.add_door(0, 0, 0, locked=False, color='red')
        door2, _ = self.add_door(1, 0, 0, locked=True, color='red')
        # Add a key to unlock the door
        key, _=self.add_object(1, 0, 'key', "red")
        ball, _=self.add_object(1, 0, 'ball', "red")
        
        #self.grid.set(2*(self.room_size-1)+self.room_size//2,self.room_size//2, Goal())

        #goal_key, _=self.add_object(1, 0, 'key', 'green') 
        #box, _=self.add_object(1, 0, 'box', 'blue')
        #box.contains = goal_key

        # make this two rooms connected via a gap 
        self.grid.set(self.room_size-1, self.room_size//2, None)
        self.grid.set(2*(self.room_size-1),self.room_size//2, None)

        self.place_agent(0, 0)
        self.agent_pos=[1, 1]
        print("agent pos",self.agent_pos)

        #self.door1 = door1
        #self.door1=door1
        self.door2=door2
        #self.box=box
        #self.goal_key=goal_key


    def step(self, action):
        obs, reward, done, info = super().step(action)
        #if self.carrying:
        #    if self.carrying.type == self.targetType:
        #        reward = self._reward()
        #        done = True
        #print(self.door1.is_open)

        #if action == self.actions.toggle:
        #    if self.door2.is_open: # and self.door2.is_open:
        #        reward = self._reward()
        #        done = True

        if self.carrying:
            if ( self.carrying.color == self.target_color and self.carrying.type == self.target_type):
                reward = self._reward() 
                done = True
            else: 
                reward = 0 
                done = True
        #print("task2", self.door1.is_locked)

        return obs, reward, done, info


register(
    id='MiniGrid-MyUnlock1-v0',
    entry_point='gym_minigrid.envs:MyUnlock1'
)
register(
    id='MiniGrid-MyUnlock2-v0',
    entry_point='gym_minigrid.envs:MyUnlock2'
)
register(
    id='MiniGrid-UpperTunel-v0',
    entry_point='gym_minigrid.envs:UpperTunel'
)

register(
    id='MiniGrid-LowerTunel-v0',
    entry_point='gym_minigrid.envs:LowerTunel'
)

register(
    id='MiniGrid-SmallUnlock-v0',
    entry_point='gym_minigrid.envs:SmallUnlock'
)


