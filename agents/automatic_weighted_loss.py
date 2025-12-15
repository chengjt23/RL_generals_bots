import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """
    自动加权 Loss 模块 (Kendall & Gal, 2018)
    公式: L_total = 0.5 * exp(-s) * L_task + 0.5 * s
    其中 s = log(sigma^2) 是可学习参数。
    """
    def __init__(self, num_losses=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始化参数 s (log variances)
        # 初始化为 0，意味着初始 sigma=1，初始权重=0.5，这是一个中性的起点
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *losses):
        """
        输入: loss1, loss2, ... (注意：输入必须是标量 Loss，不要先乘权重)
        输出: 加权后的总 Loss
        """
        loss_sum = 0
        for i, loss in enumerate(losses):
            # 获取对应的 s
            s = self.params[i]
            
            # 核心公式
            # 第一项: 1/(2*sigma^2) * Loss -> 降低高噪声任务的权重
            # 第二项: log(sigma) -> 惩罚项，防止 sigma 无限大
            loss_sum += 0.5 * torch.exp(-s) * loss + 0.5 * s
            
        return loss_sum

