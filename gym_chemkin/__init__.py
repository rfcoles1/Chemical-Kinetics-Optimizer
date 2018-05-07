import logging 
from gym.envs.registration import register 

logger = logging.getLogger(__name__)

register(id= 'chemkin-v0', entry_point = 'gym_chemkin.envs:ChemKinEnv')
