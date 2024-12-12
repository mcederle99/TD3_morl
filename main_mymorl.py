import numpy as np
import torch
import gymnasium as gym
from wrapper import TwoDimRewardWrapper
import argparse
import os

import utils_mymorl
import prioritized_buffer
import TD3_mymorl
import OurDDPG
import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env_base = gym.make(env_name, ctrl_cost_weight=1.0)
	eval_env = TwoDimRewardWrapper(eval_env_base)

	avg_reward = 0.
	omegas = np.linspace(0.0, 1.0, num=eval_episodes*10, dtype=np.float64)

	pareto_vel = []
	pareto_cost = []

	for i in range(eval_episodes*10):
		velocities = []
		actions = []
		omega = [omegas[i], 1 - omegas[i]]
		state, info = eval_env.reset(seed=seed + 100)
		done = False
		while not done:
			action = policy.select_action(np.array(state), omega)
			state, rewards, done, trunc, info = eval_env.step(action)
			reward = np.matmul(np.array(omega).T, np.array(rewards))
			velocities.append(info['x_velocity'])
			actions.append(np.matmul(action.T, action))
			done = done or trunc

			avg_reward += reward

		pareto_vel.append(np.mean(velocities))
		pareto_cost.append(np.mean(actions))

	avg_reward /= eval_episodes

	# np.save('pareto/pareto_vel_prior.npy', pareto_vel)
	# np.save('pareto/pareto_cost_prior.npy', pareto_cost)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3_mymorl")           # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v5")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=128, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--prioritized", action="store_true")
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	if args.prioritized:
		file_name = f"{args.policy}_{args.env}_{args.seed}_pr"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env_base = gym.make(args.env, ctrl_cost_weight=1.0)
	env = TwoDimRewardWrapper(env_base)

	# Set seeds
	# env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3_mymorl":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["prioritized"] = args.prioritized
		policy = TD3_mymorl.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
	else:
		policy = None

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
		evaluations = [eval_policy(policy, args.env, args.seed)]
		raise KeyboardInterrupt

	if args.prioritized:
		replay_buffer = prioritized_buffer.ReplayBuffer(2 ** 18, 0.6, state_dim, action_dim)
	else:
		replay_buffer = utils_mymorl.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]
	best_evaluation = evaluations[0]

	state, info = env.reset(seed=args.seed)
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		om = np.random.rand(1).item()
		omega = [om, 1 - om]
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state), omega)
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, rewards, done, trunc, _ = env.step(action)
		reward = np.matmul(np.array(omega).T, np.array(rewards))
		# if episode_timesteps == 200:
		# 	trunc = True
		done_bool = float(done or trunc) if episode_timesteps < 1000 else 0
		done = done or trunc

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool, omega)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, info = env.reset(seed=args.seed)
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			# om = np.random.rand(1).item()
			# omega = [om, 1 - om]

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			if args.save_model:
				np.save(f"./results/{file_name}", evaluations)
			if args.save_model and evaluations[-1] > best_evaluation:
				policy.save(f"./models/{file_name}")
				best_evaluation = evaluations[-1]
