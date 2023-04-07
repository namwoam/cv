################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'adam_da+_modified'  # name of experiment

# Model Options
model_type = 'resnet'  # 'mynet' or 'resnet18'

# Learning Options
epochs = 50           # train how many epochs
batch_size = 32           # batch size for dataloader
use_adam = True      # Adam or SGD optimizer
lr = 0.0001      # learning rate
milestones = [16, 32, 45]  # reduce learning rate at 'milestones' epochs
