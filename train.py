import torch
import torch.optim as optim
from agent import PokerActor
from critic import Critic
from replay_buffer import ReplayBuffer
from environment import PokerEnvironment

def train(config, env, actor, critic, replay_buffer, device):
    actor_optimizer = optim.Adam([p for actor_net in actor.actors for p in actor_net.parameters()], lr=config['actor_lr'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config['critic_lr'])

    for episode in range(config['episodes']):
        state = env.reset()
        done = False
        step = 0
        while not done:
            nn_index_a = step if step % 2 == 0 else step + 4
            nn_index_b = step if step % 2 == 1 else step + 4
            state_a = env.get_actor_state('A', nn_index_a)
            state_b = env.get_actor_state('B', nn_index_b)
            action_a = actor.get_action(state_a, nn_index_a)
            action_b = actor.get_action(state_b, nn_index_b)
            next_state, reward_a, reward_b, done = env.step(action_a, action_b)
            replay_buffer.push(state, [action_a, action_b], [reward_a, reward_b], next_state, done)
            state = next_state
            step += 1

            if len(replay_buffer) > config['batch_size']:
                states, actions, rewards, next_states, dones = replay_buffer.sample(config['batch_size'])
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                q_values = critic(states)
                next_q_values = critic(next_states)
                target_q = rewards + config['gamma'] * next_q_values * (1 - dones)
                critic_loss = torch.mean((q_values - target_q) ** 2)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(states).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        print(f"Episode {episode + 1}/{config['episodes']} completed")
