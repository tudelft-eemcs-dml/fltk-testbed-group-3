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
def kl_weighting(model, models_to_avg, entropies, include_federator=True):
    gp_size = len(models_to_avg)
    e_w = np.array([np.exp(e.item()) for e in entropies])

    if not include_federator:
        e_w /= sum(e_w[1:])
    else:
        e_w /= sum(e_w)

    for param in model.parameters():
        if not include_federator:
            param.data = torch.zeros(param.size() - 1)
        else:
            param.data = torch.zeros(param.size())
        gather_list = [torch.zeros(param.size()) for _ in range(gp_size)]

        for w, t in zip(e_w, gather_list):
            param.data += t * w

    return model
