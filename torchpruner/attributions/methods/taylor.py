import numpy as np
import torch
from ..attributions import _AttributionMetric


class TaylorAttributionMetric(_AttributionMetric):
    """
    Compute attributions as average absolute first-order Taylor expansion of the loss

    Reference:
    Molchanov et al., Pruning convolutional neural networks for resource efficient inference
    """

    def __init__(self, *args, signed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.signed = signed

    def run(self, module, **kwargs):
        module = super().run(module, **kwargs)
        handles = [module.register_forward_hook(self._forward_hook()),
                   module.register_backward_hook(self._backward_hook())
                   ]
        self.run_all_forward_and_backward()
        attr = module._tp_taylor
        result = self.aggregate_over_samples(attr)
        delattr(module, "_tp_taylor")
        for h in handles:
            h.remove()
        return result

    @staticmethod
    def _forward_hook():
        def _hook(module, _, output):
            with torch.no_grad():
                # output.permute(0,2,1)
                # output.contiguous()
                module._tp_activation = output.detach().clone()
        return _hook

    def _backward_hook(self):
        def _hook(module, _, grad_output):
            taylor = -1. * (grad_output[0] * module._tp_activation)
            # print("talor0:{}".format(taylor.shape))
            if len(taylor.shape) > 2:
                taylor = taylor.flatten(2).sum(1)
                # print("talor1:{}".format(taylor.shape))
            if self.signed is False:
                taylor = taylor.abs()
            if not hasattr(module, "_tp_taylor"):
                module._tp_taylor = taylor.detach().cpu().numpy()
                # print('yes')
            else:
                module._tp_taylor = np.concatenate((module._tp_taylor, taylor.detach().cpu().numpy()), 0)
                # print('no')
                # print(module._tp_taylor.shape)
        return _hook




