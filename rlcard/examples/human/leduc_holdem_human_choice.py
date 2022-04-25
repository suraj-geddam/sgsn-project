''' A toy example of playing against pretrianed AI of our choice on Leduc Hold'em
'''

import argparse
import os

import rlcard
import torch
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.utils import print_card

def load_model(model_path, env=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    else:  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    
    return agent

parser = argparse.ArgumentParser("Evaluation example in RLCard")
parser.add_argument(
        '--model',
        default='experiments/leduc_holdem_dqn_result/model.pth'
)
args = parser.parse_args()

# Make environment
env = rlcard.make('leduc-holdem')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
human_agent = HumanAgent(env.num_actions)
other_agent = load_model(args.model, env, device)
env.set_agents([
    human_agent,
    other_agent,
])

print(">> Leduc Hold'em pre-trained model at", args.model)

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     Other Agent    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
