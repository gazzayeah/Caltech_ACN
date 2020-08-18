# import parser package
import argparse
from datetime import datetime
import pytz
from gym_EV.envs.reward_functions import *
# define timezone
timezone = pytz.timezone('America/Los_Angeles')

# ---------------------------------------------------------------------------------
#  EV-gym ACN Data
#  
#  Below are all arguments required for defining Training and
#  testing data from ACN Database.
#
# ---------------------------------------------------------------------------------
# set parser that defines invariant global variables
parser = argparse.ArgumentParser(description='ACN Data Settings')

parser.add_argument('--START', type = datetime, default = timezone.localize(datetime(2018, 5, 1)),
                    help='starting date of training data (default: 2018, 5, 1)')

parser.add_argument('--END_TRAIN', type = datetime, default = timezone.localize(datetime(2018, 6, 15)),
                    help='ending date of training data (default: 2018, 6, 15)')

parser.add_argument('--END_TEST', type = datetime, default = timezone.localize(datetime(2018, 6, 22)),
                    help='ending date of testing data (default: 2018, 6, 22)')

parser.add_argument('--PHASE_SELECT', type = bool, default = False,
                    help='decide to follow real ACN phase selection if True (default: True)')

# pack up ACN data parsers
dataArgs = parser.parse_args()




# ---------------------------------------------------------------------------------
#  EV Gym Parameters
#  
#  Below are all arguments required for EV Gym.
#
#  Structure of reward function is determined.
# ---------------------------------------------------------------------------------

# set parser that defines invariant global variables
parser = argparse.ArgumentParser(description='EV-gym Settings')

# set ev env id to parser, we call ev env by this argument
parser.add_argument('--ENV_NAME', default="EV-v0",
                    help='name of the environment to run')

# set seed value to random generator, this is a tuning parameter for sepecific algorithms
parser.add_argument('--SEED', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')

# start with small EVSE network
parser.add_argument('--MAX_EV', type=int, default=21, metavar='N',
                    help='maximum EVSEs in the network (default: 30)')

# maximum power assignment for individual EVSE
parser.add_argument('--MAX_RATE', type=float, default=6, metavar='G',
                    help='maximum assignable rates for a single EVSE (default: 5)')

# oversubscription level from {1, 2, 3 ...}
parser.add_argument('--INTENSITY', type=int, default=1, metavar='N', 
                    help='level of oversubsription for the charging network (default: 2)')

# ACN generator choose random date or not used in gym
parser.add_argument('--RANDOM_DATE', type=bool, default=False, metavar='B', 
                    help='determine if ACN data reset with a random date in EV-gym (default: False)')

# pack up EV-gym parsers
gymArgs = parser.parse_args()

# Define reward functions
REWARD_FUNCTION = [RewardComponent(l2_norm_reward, 2), RewardComponent(deadline_penalty, 1)]



# ---------------------------------------------------------------------------------
#  EV Gym Parameters
#  
#  Below are all arguments required for EV Gym.
#
#  Structure of reward function is determined.
# ---------------------------------------------------------------------------------

# set parser that defines invariant global variables
parser = argparse.ArgumentParser(description='Network Settings')

# maximum power assignment for whole network
# less than MAX_EV*MAX_RATE
parser.add_argument('--MAX_CAPACITY', type=float, default=30, metavar='G',
                    help='maximum assignable rates for whole network (default: 40)')

# turning ratio of step-down transformer
parser.add_argument('--TURN_RATIO', type=int, default=4, metavar='N', 
                    help='turning ratio of step-down transformer (default: 4)')

# phase partition: deploying specified number of EVSEs to each of the phase line
parser.add_argument('--PHASE_PARTITION', type=list, default=[7, 7], metavar='L', 
                    help='constraint type of charging network(default: [2, 2])')

# constraint type: 'SOC'(three phase) or 'LINEAR'(single phase)
parser.add_argument('--CONSTRAINT_TYPE', type=str, default='SOC', metavar='S', 
                    help='constraint type of charging network(default: SOC)')


# pack up EV-gym parsers
netArgs = parser.parse_args()



# ---------------------------------------------------------------------------------
#  Experiment Parameters
#  
#  Below are all arguments required for experiment in main
#
# ---------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Experiment Settings')

# set maximum iteration for learning episodes (max 1000000 episodes by default)
parser.add_argument('--TRAIN_EPISODES', type=int, default=1, metavar='N',
                    help='maximum number of episodes (default: 1000)')

# set maximum iteration for learning episodes (max 1000000 episodes by default)
parser.add_argument('--TEST_EPISODES', type=int, default=100, metavar='N',
                    help='maximum number of episodes (default: 100)')

# if true, process evaluation for every 10 episode
parser.add_argument('--EVAL_PERIOD', type=int, default=10, metavar='N',
                    help='period between two evaluation happens with episode as the unit (default:10)')

# if true, process evaluation for every 10 episode
parser.add_argument('--EVAL', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')

# pack up experiments parsers
expArgs = parser.parse_args()