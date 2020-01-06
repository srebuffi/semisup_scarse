from .ResNet import ResNet18
from .TempEns import TempEns
from .RevNet import RevNet18


def load_net(network, classes=4):
    # Load model
    if network == 'ResNet-18':
        model = ResNet18(classes)
    elif network == 'TempEns':
        model = TempEns(output_dim=classes)
    elif network == 'RevNet-18':
        model = RevNet18(classes)
    else:
        raise NotImplementedError('Network {0} not implemented'.format(network))
    return model
