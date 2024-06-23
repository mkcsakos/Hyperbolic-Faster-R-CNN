from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner import Runner
from mmdet.models import PeBusePenaltyLoss, WeightedBusePenaltyLoss

@HOOKS.register_module()
class PeBusePenaltyLossResetVariablesHook(Hook):
    def before_train_epoch(self, runner: Runner) -> None:
        runner.logger.info('Resetting variables after epoch')

        for module in runner.model.modules():
            if isinstance(module, PeBusePenaltyLoss):
                module.reset_after_epoch()

                runner.logger.info('Variables reset')



@HOOKS.register_module()
class WeightedPeBusePenaltyLossResetVariablesHook(Hook):
    def before_train_epoch(self, runner: Runner) -> None:
        runner.logger.info('Resetting variables after epoch')

        for module in runner.model.modules():
            if isinstance(module, WeightedBusePenaltyLoss):
                module.reset_after_epoch()

                runner.logger.info('Variables reset')
