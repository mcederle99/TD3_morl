import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from labml_helpers.schedule import Piecewise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.enc = nn.Linear(2, 64)
		self.l2 = nn.Linear(256+64, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state, omega):
		state = F.relu(self.l1(state))
		omega = F.relu(self.enc(omega))
		so = torch.cat([state, omega], 1)
		a = F.relu(self.l2(so))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.enc1 = nn.Linear(2, 64)
		self.l2 = nn.Linear(256+64, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.enc2 = nn.Linear(2, 64)
		self.l5 = nn.Linear(256+64, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action, omega):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		omega1 = F.relu(self.enc1(omega))
		q1 = torch.cat([q1, omega1], 1)
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		omega2 = F.relu(self.enc2(omega))
		q2 = torch.cat([q2, omega2], 1)
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action, omega):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		omega1 = F.relu(self.enc1(omega))
		q1 = torch.cat([q1, omega1], 1)
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		prioritized=False
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		self.prioritized = prioritized

		if self.prioritized:
			self.prioritized_replay_beta = Piecewise(
				[
					(0, 0.4),
					(1e6, 1)
				], outside_value=1)
			self.loss = nn.MSELoss(reduction='none')

	def select_action(self, state, omega):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		omega = torch.FloatTensor(omega).unsqueeze(dim=0).to(device)
		return self.actor(state, omega).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		if self.prioritized:
			beta = self.prioritized_replay_beta(self.total_it)
			state, action, next_state, reward, not_done, omegas, weights, indexes = replay_buffer.sample(batch_size,
																										 beta)
		else:
			state, action, next_state, reward, not_done, omegas = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state, omegas) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action, omegas)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward.unsqueeze(dim=-1) + not_done.unsqueeze(dim=-1) * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action, omegas)

		# Compute critic loss
		if self.prioritized:
			td_errors = current_Q1 - target_Q
			critic_loss = self.loss(current_Q1, target_Q) + self.loss(current_Q2, target_Q)
			critic_loss = torch.mean(weights * critic_loss)
			# Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
			new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
			# Update replay buffer priorities
			replay_buffer.update_priorities(indexes, new_priorities)
		else:
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state, omegas), omegas).mean()

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
		self.actor_target = copy.deepcopy(self.actor)
		