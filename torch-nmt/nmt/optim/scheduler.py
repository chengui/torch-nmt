import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, d_model=-1, **kw):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, **kw)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        if self.d_model <= 0:
            base_lrs = self.base_lrs
        else:
            new_base = np.power(self.d_model, -0.5)
            base_lrs = [new_base] * len(self.base_lrs)
        gamma = np.min([
            np.power(self.last_epoch, -0.5),
            np.power(self.warmup_steps, -1.5) * self.last_epoch])
        return [lr * gamma for lr in base_lrs]


if __name__ == '__main__':
    from torch import optim
    from nmt.model import TransformerSeq2Seq
    from matplotlib import pyplot as plt

    model = TransformerSeq2Seq(101, 102)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = NoamScheduler(optimizer, 400, 512)
    x, y = list(range(10000)), []
    for epoch in range(len(x)):
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_lr()
        y.append(lr[0])

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.plot(x, y)
    plt.savefig('1.png')

