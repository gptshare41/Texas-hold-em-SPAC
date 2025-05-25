import argparse
import json
import os
import torch
from environment import PokerEnvironment
from agent import PokerActor
from critic import Critic
from replay_buffer import ReplayBuffer
from train import train

def main():
    parser = argparse.ArgumentParser(description="Poker SPAC Training")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode to run')
    args = parser.parse_args()

    with open('config/config.json', 'r') as f:
        config = json.load(f)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_devices']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PokerEnvironment(config)
    actor = PokerActor(config).to(device)
    critic = Critic(config['state_dim'], config['hidden_dim']).to(device)
    replay_buffer = ReplayBuffer(config['buffer_size'], config['state_dim'])

    if args.mode == 'train':
        train(config, env, actor, critic, replay_buffer, device)
    elif args.mode == 'evaluate':
        # 평가 모드 (미구현)
        print("Evaluation mode is not implemented yet.")

if __name__ == "__main__":
    main()
