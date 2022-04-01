# DQN vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'random'
# NFSP vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_nfsp_result/model.pth' 'random'
# CFR vs random
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_cfr_result/cfr_model' 'random'
# DQN vs NFSP
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'experiments/leduc_holdem_nfsp_result/model.pth'
# DQN vs CFR
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_dqn_result/model.pth' 'experiments/leduc_holdem_cfr_result/cfr_model'
# NFSP vs CFR
python rlcard/examples/evaluate.py --cuda 0 --models 'experiments/leduc_holdem_nfsp_result/model.pth' 'experiments/leduc_holdem_cfr_result/cfr_model'
