import numpy as np
import torch


def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params


# TODO: check that it is indeed correct
def kl_weighting(model, models_to_avg, entropies):
    print('calculate KL divergence')
    e_w = np.array([np.exp(e.item()) for e in entropies])
    e_w /= sum(e_w[1:])

    for idx, param in enumerate(model.parameters()):
        param.data = torch.zeros(param.size())

        for model in models_to_avg:
            for w, t in zip(e_w, list(model.parameters())[idx]):
                param.data += t * w

    return model
