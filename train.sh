# train cfr on leduc holdem
python rlcard/examples/run_cfr.py
# train dqn on leduc holdem
python rlcard/examples/run_rl.py --cuda 0
# train nfsp on leduc holdem
python rlcard/examples/run_rl.py --cuda 0 --algorithm nfsp --log_dir 'experiments/leduc_holdem_nfsp_result/'
# train ppo on leduc holdem
python rlcard/examples/run_ppo.py --cuda 0