''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents import PPOAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    agent = PPOAgent(state_dim=env.state_shape[0][0],
        action_dim=env.num_actions,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=20,
        eps_clip=0.2,
        has_continuous_action_space=False,
        action_std_init=0.4,
        device=device,
        )
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that PPO always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                reward = ts[2]
                done = ts[4]
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
            
            lrew = len(agent.buffer.rewards)
            agent.buffer.states = agent.buffer.states[-lrew:]
            agent.buffer.actions = agent.buffer.actions[-lrew:]
            agent.buffer.logprobs = agent.buffer.logprobs[-lrew:]
            #print("States:", len(agent.buffer.states), "Rewards:", lrew)

            if episode % 10 == 9:
                agent.update()

            #agent.buffer.clear()

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

            #agent.buffer.clear()

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, "PPO")

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_ppo_result/',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
