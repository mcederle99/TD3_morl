import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 2))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim = state_dim
		self.action_dim = action_dim

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		omegas = np.random.rand(32, 1)
		om = np.concatenate((omegas, 1-omegas), axis=1)

		aug_state = np.zeros((batch_size*32, self.state_dim+2))
		aug_next_state = np.zeros((batch_size*32, self.state_dim+2))
		aug_action = np.zeros((batch_size*32, self.action_dim))
		aug_not_done = np.zeros((batch_size*32, 1))
		aug_reward = np.zeros((batch_size * 32, 2))
		aug_omegas = np.zeros((batch_size * 32, 2))

		for i in range(batch_size*32):
			aug_state[i] = np.concatenate(([self.state[ind[i % batch_size]]], [om[i % 32]]), axis=1)
			aug_next_state[i] = np.concatenate(([self.next_state[ind[i % batch_size]]], [om[i % 32]]), axis=1)
			aug_action[i] = self.action[ind[i % batch_size]]
			aug_not_done[i] = self.not_done[ind[i % batch_size]]
			aug_reward[i] = self.reward[ind[i % batch_size]]
			aug_omegas[i] = om[i % 32]

		return (
			torch.FloatTensor(aug_state).to(self.device),
			torch.FloatTensor(aug_action).to(self.device),
			torch.FloatTensor(aug_next_state).to(self.device),
			torch.FloatTensor(aug_reward).to(self.device),
			torch.FloatTensor(aug_not_done).to(self.device),
			torch.FloatTensor(aug_omegas).to(self.device)
		)
