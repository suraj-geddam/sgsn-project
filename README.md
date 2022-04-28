# COMS 4995 (Deep Learning) Project
## RL Models for Poker

Make sure you have torch installed.
Then, navigate to the rlcard directory and install RLcard by the following command:
```
sudo python setup.py install
```
Then, to run the training and evaluation script, use:
```
bash train_and_evaluate.sh
```
For only training, you can use `bash train.sh` and if training has already been done, evaluation can be run by `bash evaluate.sh`.

After the models have been trained, if you wish to play a game against one of them (located in rlcard/examples/human):
```
python leduc_holdem_human_choice.py --model /path/to/model
```

Thanks to the authors of RLCard and this implementation of PPO!

RLCard source code from https://github.com/datamllab/rlcard

PPO implementation from https://github.com/nikhilbarhate99/PPO-PyTorch
