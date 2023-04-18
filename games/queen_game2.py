from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax

from games.env import Enviroment
from utils import select_tree
from jax import vmap, jit
from jax.lax import while_loop

class Ripper(pax.Module):
    
    num_cols: int = 9
    num_rows: int = 8
    
    def __init__(self):
        super().__init__()
        self.all_coords = jnp.stack([jnp.repeat(jnp.arange(self.num_rows),repeats=self.num_cols),jnp.tile(jnp.arange(self.num_cols),reps=self.num_rows)],axis=0)

    def __call__(self, board: jnp.array, coords: jnp.array, new_coords: jnp.array):
        translation = new_coords - coords
        coords_tile = jnp.tile(coords,reps=(self.num_cols*self.num_rows,1)).transpose()
        new_coords_tile = jnp.tile(new_coords,reps=(self.num_cols*self.num_rows,1)).transpose()
        temp_translation = self.all_coords - coords_tile
        temp_translation2 = new_coords_tile - self.all_coords
        #print(temp_translation.shape, temp_translation2.shape)

        def vacuuming(trans, temp_trans,temp_trans2):
            det = jnp.linalg.det(jnp.stack([trans, temp_trans]))
            dire = jnp.logical_and(jnp.dot(temp_trans, trans) >= 0, jnp.dot(temp_trans2,trans) >= 0)
            return jnp.logical_and(dire, jnp.abs(det)<=0.01).astype(jnp.int8)
        
        trace = vmap(vacuuming, in_axes=(None,1,1))(translation, temp_translation,temp_translation2)
        trace = trace.reshape((self.num_rows,self.num_cols))

        return board - trace
    
class ActionMasker(pax.Module):
    num_cols: int = 9
    num_rows: int = 8
    def __init__(self):
        super().__init__()
        self.movements = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=jnp.int8)

    def explore(self,board, x, y, d_x, d_y, mask):
        def cond(val):
            state, x, y, d_x, d_y, conds, board, mask = val
            del x, y, board, mask
            d_conds = jnp.array([d_x > -1 , d_x < 1, d_y > -1, d_y < 1], dtype=jnp.bool_)
            conds = jnp.logical_or(d_conds,conds)
            conds = jnp.logical_and(conds[:2],conds[2:])
            conds = jnp.logical_and(conds[0],conds[1])
            return jnp.logical_and(conds, state!=-1)

        def probe(val):
            state, x, y, d_x, d_y, conds, board, mask = val
            x += d_x
            y += d_y
            state = board[x, y]
            mask = jnp.where(state != -1,mask.at[x,y].set(1),mask)
            conds = jnp.array([x > 0, x < (board.shape[0] - 1), y > 0, y < (board.shape[1] - 1)], dtype=jnp.bool_)
            val = state, x, y, d_x, d_y, conds, board, mask
            return val

        conds = jnp.array([x > 0, x < (board.shape[0] - 1), y > 0, y < (board.shape[1] - 1)], dtype=jnp.bool_)
        container = 0, x, y, d_x, d_y, conds, board, mask
        state, x, y, d_x, d_y, conds, board, mask = while_loop(cond,probe,container)

        return mask

    def __call__(self, board, coords):
        n_board = jnp.reshape(board,(self.num_rows,self.num_cols))
        mask = jnp.zeros_like(n_board)
        x = coords[0]
        y = coords[1]

        def oct_explore(n_board, x, y, movement):
            return self.explore(n_board, x, y, movement[0], movement[1], mask)

        mask = vmap(oct_explore,in_axes=(None,None,None,0))(n_board,x,y,self.movements).sum(axis=0)

        return mask
    
    
class FlightOfQueenGame(Enviroment):
    """FOQ game environment"""

    board: chex.Array
    who_play: chex.Array
    terminated: chex.Array
    recent_boards: chex.Array  # a list of recent positions
    winner: chex.Array
    num_cols: int = 9
    num_rows: int = 8
    num_recent_positions: int = 4

    def __init__(self, num_cols: int = 9, num_rows: int = 8):
        super().__init__()
        self.masker = ActionMasker()
        self.ripper = Ripper()
        self.board = jnp.zeros((num_rows, num_cols), dtype=jnp.int32)
        self.recent_boards = jnp.stack([self.board] * self.num_recent_positions)
        self.who_play = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.coords = jnp.array([[0,0],[7,8]],dtype=jnp.int32)
        self.all_coords = jnp.stack([jnp.repeat(jnp.arange(self.num_rows),repeats=self.num_cols),jnp.tile(jnp.arange(self.num_cols),reps=self.num_rows)],axis=0)
        self.reset()

    def num_actions(self):
        return self.num_cols * self.num_rows
    
    def duo_mask(self, whom):
        mask = jit(self.masker)(self.board, self.coords[whom])
        mask = mask - jit(self.masker)(self.board, self.coords[(whom + 1) % 2])
        return  mask.flatten()

    def invalid_actions(self) -> chex.Array:
        return self.duo_mask(self.who_play) <= 0

    def reset(self):
        self.board = jnp.zeros((self.num_rows, self.num_cols), dtype=jnp.int32)
        self.recent_boards = jnp.stack([self.board] * self.num_recent_positions)
        self.who_play = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["FlightOfQueenGame", chex.Array]:
        """One step of the game.
        An invalid move will terminate the game with reward -1.
        """
        invalid_move = self.duo_mask(self.who_play)[action] <= 0
        new_coords = self.all_coords[:,action]
        #self.board = self.board.at[self.coords[self.who_play, 0]*9+self.coords[self.who_play, 1]].set(-1)
        board_ = self.ripper(self.board, self.coords[self.who_play],new_coords)
        board_ = jnp.clip(board_,-1,2)
        self.coords = self.coords.at[self.who_play].set(new_coords)
        self.board = select_tree(self.terminated, self.board, board_)
        self.recent_boards = jnp.concatenate((self.board[None], self.recent_boards[:-1]))
        
        self.winner = jnp.where((self.duo_mask(self.who_play)<=0).sum() == 0,-1,0)
        reward_ = jnp.where(self.who_play == 0,self.winner,-self.winner)
        # increase column counter
        self.who_play = 1 - self.who_play
        self.count = self.count + 1
        self.terminated = jnp.logical_or(self.terminated, reward_ != 0)
        self.terminated = jnp.logical_or(
            self.terminated, self.count >= self.num_cols * self.num_rows
        )
        self.terminated = jnp.logical_or(self.terminated, invalid_move)
        reward_ = jnp.where(invalid_move, -1.0, reward_)
        return self, reward_

    def render(self) -> None:
        """Render the game on screen."""
        #board = jnp.reshape(self.board,(self.num_rows,self.num_cols))
        board = self.board
        board = board.at[self.coords[self.who_play, 0], self.coords[self.who_play, 1]].set(1)
        board = board.at[self.coords[1-self.who_play, 0], self.coords[1-self.who_play, 1]].set(2)
        #print(self.observation())
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if board[row, col].item() == 1:
                    print("X", end=" ")
                elif board[row, col].item() == -1:
                    print("O", end=" ")
                elif board[row, col].item() == 2:
                    print("A", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()
        
    def neutral_obs(self, whom) -> chex.Array:
        board = jnp.zeros_like(self.board)
        board = board.at[self.coords[whom,0],self.coords[whom,1]].set(1)
        board = board.at[self.coords[1-whom,0],self.coords[1-whom,1]].set(-1)[None]
        coord_visual = self.duo_mask(whom).reshape((self.num_rows,self.num_cols))[None]
        return jnp.concatenate([self.recent_boards,board,coord_visual])

    def observation(self) -> chex.Array:
        #return jnp.reshape(board, (self.num_rows, self.num_cols))
        return self.neutral_obs(0)

    def canonical_observation(self) -> chex.Array:
        return self.neutral_obs(self.who_play)

    def is_terminated(self):
        return self.terminated

    def max_num_steps(self) -> int:
        return self.num_cols * self.num_rows

    def symmetries(self, state, action_weights):
        action = action_weights.reshape((self.num_rows, self.num_cols))
        out = []
        #out.append((state, action_weights))
        for i in [0,2]:
            rotated_state = np.rot90(state, k=i, axes=(0, 1))
            rotated_action = np.rot90(action, k=i, axes=(0, 1))
            out.append((rotated_state, rotated_action.reshape((-1,))))

            flipped_state = np.fliplr(rotated_state)
            flipped_action = np.fliplr(rotated_action)
            out.append((flipped_state, flipped_action.reshape((-1,))))

        return out
    
    
if __name__ == "__main__":
    game = FlightOfQueenGame()
    game.render()
    game, reward = game.step(50)
    game.render()