import torch
import copy

import detectron2.utils.comm as comm
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.best_loss = float('inf')
        self.weights = None

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)

                # save best weights
                if losses_reduced < self.best_loss:
                    self.best_loss = losses_reduced
                    self.weights = copy.deepcopy(self.trainer.model.state_dict())
