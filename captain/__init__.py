__version__ = "0.1.0"
__citation__ = """
    Silvestro D, Goria S, Sterner T, Antonelli A. 
    Improving biodiversity protection through artificial intelligence
    (2021)
"""

from . import biodivsim
from . import biodivinit
from . import algorithms
from . import agents
from .biodivinit import PhyloGenerator
from .biodivinit import SimulatorInit

from .biodivsim.CellClass import *
from .biodivsim.StateInitializer import *
from .biodivsim.BioDivEnv import *
from .biodivsim.DisturbanceGenerator import *
from .biodivsim.ClimateGenerator import *
from .biodivinit.PhyloGenerator import *
from .algorithms.geneticStrategies import *
from .algorithms.runOptimizedPolicy import *
from .biodivsim.EmpiricalBioDivEnv import *
from .biodivinit.SimulatorInit import *
from .plot.plot_env import *
from .biodivsim.EmpiricalGrid import *
from .algorithms.runPolicyEmpirical import *
from .biodivsim.ConservationTargets import *
