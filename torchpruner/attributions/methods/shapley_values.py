import torch
import numpy as np
import logging
from ..attributions import _AttributionMetric
from itertools import combinations, permutations
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO )
logger = logging.getLogger(__name__)

class ShapleyAttributionMetric(_AttributionMetric):
    """
    Compute attributions as approximate Shapley values using sampling.
    """

    def __init__(self, *args, sv_samples=5, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.samples = sv_samples
        self.mask_indices = []

    def run(self, module,module2=None, sv_samples=None, **kwargs):
        module = super().run(module, **kwargs)
        sv_samples = sv_samples if sv_samples is not None else self.samples
        if hasattr(self.model, "forward_partial"):
            result = self.run_module_with_partial(module, sv_samples)
        else:
            logging.warning("Consider adding a 'forward_partial' method to your model to speed-up Shapley values "
                            "computation")
            result = self.run_module(module, module2,sv_samples)
        return result

    def run_module_with_partial(self, module, sv_samples):    
        """
        Implementation of Shapley value monte carlo sampling for models
        that provides a `forward_partial` function. This is significantly faster
        than run_module(), as it only runs the forward pass on the necessary modules.
        """
        d = len(self.data_gen.dataset)
        sv = None
        permutations = None
        c = 0

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.data_gen): # val_loader
                x, y = x.to(self.device), y.to(self.device) # y:类型 x：图片数据
                original_z, _ = self.run_forward_partial(x, to_module=module) # original_z 初始图片部分前向计算后的输出
                _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module) # 验证输出后的loss
                n = original_z.shape[1]  # prunable dimension
                if permutations is None:
                    # Keep the same permutations for all batches
                    permutations = [np.random.permutation(n) for _ in range(sv_samples)]
                if sv is None:
                    sv = np.zeros((d, n))

                for j in range(sv_samples):
                    loss = original_loss.detach().clone()
                    z = original_z.clone().detach()

                    for i in permutations[j]:
                        z.index_fill_(1, torch.tensor(np.array([i])).long().to(self.device), 0.0)
                        _, new_loss = self.run_forward_partial(z, y_true=y, from_module=module)
                        delta = new_loss - loss
                        n = delta.shape[0]
                        sv[c:c+n, i] += (delta / sv_samples).squeeze().detach().cpu().numpy()
                        loss = new_loss
                c += n

            return self.aggregate_over_samples(sv)

    def run_module(self, module, module2,samples):   #剪枝4head
        """
        Implementation of Shapley value monte carlo sampling.
        No further changes to the model are necessary but this can be quite slow.
        See run_module_with_partial() for a faster version that uses partial evaluation.
        """
        with torch.no_grad():
            self.mask_indices = []
            handle1 = module.register_forward_hook(self._forward_hook_1())
            # handle2 = module2.register_forward_hook(self._forward_hook_2())
            original_loss = self.run_all_forward()
            n = module._tp_prune_dim  # output dimension
            sv = np.zeros((original_loss.shape[0], n))

            for j in range(samples):
                # print (f"Sample {j}")
                cal_sha_count = 0
                # self.mask_indices = []
                # head_mask_indices = []
                loss = original_loss.detach().clone()
                for i in range(495):
                    self.mask_indices = []
                    head_mask_indices = []
                    head_mask_indices = list(list(combinations([0,1, 2, 3,4,5,6,7,8,9,10,11], 4))[i])
                    for ii in range(4):
                          head_count_value = head_mask_indices[ii]*64 + 64
                          for iii in range(head_count_value-64,head_count_value,1):
                            self.mask_indices.append(iii)
                    cal_sha_count += 1
                    if cal_sha_count%200 ==0:
                        logger.info("cal_sha_count:%d",(cal_sha_count))
            # print("cal_sha_count:{}",cal_sha_count/3072)
                    new_loss = self.run_all_forward()
                    sv[:, i] += ((new_loss - loss) / samples).squeeze().detach().cpu().numpy()
                # loss = new_loss

            handle1.remove()
            # handle2.remove()
            return self.aggregate_over_samples(sv)
            
    # def run_module(self, module, module2,samples):   #剪枝8head
    #     """
    #     Implementation of Shapley value monte carlo sampling.
    #     No further changes to the model are necessary but this can be quite slow.
    #     See run_module_with_partial() for a faster version that uses partial evaluation.
    #     """
    #     with torch.no_grad():
    #         self.mask_indices = []
    #         handle1 = module.register_forward_hook(self._forward_hook_1())
    #         # handle2 = module2.register_forward_hook(self._forward_hook_2())
    #         original_loss = self.run_all_forward()
    #         n = module._tp_prune_dim  # output dimension
    #         sv = np.zeros((original_loss.shape[0], n))

    #         for j in range(samples):
    #             # print (f"Sample {j}")
    #             cal_sha_count = 0
    #             # self.mask_indices = []
    #             # head_mask_indices = []
    #             loss = original_loss.detach().clone()
    #             for i in range(495):
    #                 self.mask_indices = []
    #                 head_mask_indices = []
    #                 head_mask_indices = list(list(combinations([0,1, 2, 3,4,5,6,7,8,9,10,11], 8))[i])
    #                 for ii in range(8):
    #                       head_count_value = head_mask_indices[ii]*64 + 64
    #                       for iii in range(head_count_value-64,head_count_value,1):
    #                         self.mask_indices.append(iii)
    #                 cal_sha_count += 1
    #                 if cal_sha_count%200 ==0:
    #                     logger.info("cal_sha_count:%d",(cal_sha_count))
    #         # print("cal_sha_count:{}",cal_sha_count/3072)
    #                 new_loss = self.run_all_forward()
    #                 sv[:, i] += ((new_loss - loss) / samples).squeeze().detach().cpu().numpy()
    #             # loss = new_loss

    #         handle1.remove()
    #         # handle2.remove()
    #         return self.aggregate_over_samples(sv)
        
    def run_module(self, module, module2,samples): ##剪枝线性层用的
        """
        Implementation of Shapley value monte carlo sampling.
        No further changes to the model are necessary but this can be quite slow.
        See run_module_with_partial() for a faster version that uses partial evaluation.
        """
        with torch.no_grad():
            self.mask_indices = []
            handle1 = module.register_forward_hook(self._forward_hook_1())
            # handle2 = module2.register_forward_hook(self._forward_hook_2())
            original_loss = self.run_all_forward()
            n = module._tp_prune_dim  # output dimension
            sv = np.zeros((original_loss.shape[0], n))

            for j in range(samples):
                # print (f"Sample {j}")
                cal_sha_count = 0
                self.mask_indices = []
                loss = original_loss.detach().clone()
                for i in np.random.permutation(n):
                    # self.mask_indices=[]        #####
                    self.mask_indices.append(i)
                    cal_sha_count += 1
                    if cal_sha_count%1000 ==0:
                        logger.info("cal_sha_count:%d",(cal_sha_count))
                        # print("cal_sha_count:{}",cal_sha_count/3072)
                    new_loss = self.run_all_forward()
                    sv[:, i] += ((new_loss - loss) / samples).squeeze().detach().cpu().numpy()
                    loss = new_loss

            handle1.remove()
            # handle2.remove()
            return self.aggregate_over_samples(sv)

    def _forward_hook_1(self):
        def _hook(module, _, output):
            module._tp_prune_dim = output.shape[2]
            return output.index_fill_(
                2, torch.tensor(self.mask_indices).long().to(self.device), 0.0,
            )
        return _hook
    
    # def _forward_hook_2(self):
    #     def _hook(module2, input, _):
    #         # input=np.array(input).astype(float)
    #         # input=torch.from_numpy(input)
    #         # input = torch.as_tensor(input)
    #         input= torch.tensor([item.cpu().detach().numpy() for item in input]).cuda()

    #         module2._tp_prune_dim = input.shape[3] 
    #         # return input.index_fill_(
    #         #     1, torch.tensor(self.mask_indices).long().to(self.device), 0.0,
    #         # )
    #         input.index_fill_(
    #         1, torch.tensor(self.mask_indices).long().to(self.device), 0.0,
    #         )
    #     return _hook


    # def run_module(self, module, module2,samples):
    #     """
    #     Implementation of Shapley value monte carlo sampling.
    #     No further changes to the model are necessary but this can be quite slow.
    #     See run_module_with_partial() for a faster version that uses partial evaluation.
    #     """
    #     with torch.no_grad():
    #         self.mask_indices = []
    #         handle = module.register_forward_hook(self._forward_hook())
    #         original_loss = self.run_all_forward()
    #         n = module._tp_prune_dim  # output dimension
    #         sv = np.zeros((original_loss.shape[0], n))

    #         for j in range(samples):
    #             # print (f"Sample {j}")
    #             self.mask_indices = []
    #             loss = original_loss.detach().clone()
    #             for i in np.random.permutation(n):
    #                 self.mask_indices.append(i)
    #                 new_loss = self.run_all_forward()
    #                 sv[:, i] += ((new_loss - loss) / samples).squeeze().detach().cpu().numpy()
    #                 loss = new_loss

    #         handle.remove()
    #         return self.aggregate_over_samples(sv)

    # def _forward_hook(self):
    #     def _hook(module, _, output):
    #         module._tp_prune_dim = output.shape[2]
    #         return output.index_fill_(
    #             1, torch.tensor(self.mask_indices).long().to(self.device), 0.0,
    #         )

    #     return _hook