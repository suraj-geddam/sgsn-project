# DQN vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'random'
# NFSP vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_nfsp_result/model.pth' 'random'
# CFR vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_cfr_result/cfr_model' 'random'
# PPO vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_ppo_result/model.pth' 'random'
# DQN vs NFSP
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'experiments/leduc_holdem_nfsp_result/model.pth'
# DQN vs CFR
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'experiments/leduc_holdem_cfr_result/cfr_model'
# NFSP vs CFR
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_nfsp_result/model.pth' 'experiments/leduc_holdem_cfr_result/cfr_model'
# DQN vs PPO
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'experiments/leduc_holdem_ppo_result/model.pth'
# NFSP vs PPO
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_nfsp_result/model.pth' 'experiments/leduc_holdem_ppo_result/model.pth'
# CFR vs PPO
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_cfr_result/cfr_model' 'experiments/leduc_holdem_ppo_result/model.pth'