# Attack plugins package
#
# Import all plugins here to ensure they are registered with AttackRegistry
# when the plugins package is imported.

from . import blackbox_loss
from . import grad_cosine
from . import grad_diff
from . import grad_norm
from . import loss_series
from . import avg_cosine
from . import fedmia_i
from . import fedmia_ii

from .blackbox_loss import BlackboxLossPlugin
from .grad_cosine import GradCosinePlugin
from .grad_diff import GradDiffPlugin
from .grad_norm import GradNormPlugin
from .loss_series import LossSeriesPlugin
from .avg_cosine import AvgCosinePlugin
from .fedmia_i import FedMIAIPlugin
from .fedmia_ii import FedMIAIIPlugin

__all__ = [
    'BlackboxLossPlugin',
    'GradCosinePlugin',
    'GradDiffPlugin',
    'GradNormPlugin',
    'LossSeriesPlugin',
    'AvgCosinePlugin',
    'FedMIAIPlugin',
    'FedMIAIIPlugin',
]
