import math
import paddle


def piecewise_decay(net, device_num, total_images, batch_size, lr,
                    momentum_rate, l2_decay, step_epochs):
    step = int(math.ceil(float(total_images) / (batch_size * device_num)))
    bd = [step * e for e in step_epochs]
    lr = [lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        parameters=net.parameters(),
        learning_rate=learning_rate,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay))
    return optimizer, learning_rate


def cosine_decay(net, device_num, total_images, batch_size, lr, momentum_rate,
                 l2_decay, num_epochs):
    step = int(math.ceil(float(total_images) / (batch_size * device_num)))
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=lr, T_max=step * num_epochs, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        parameters=net.parameters(),
        learning_rate=learning_rate,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay))
    return optimizer, learning_rate


def create_optimizer(net, lr_strategy, device_num, total_images, batch_size,
                     lr, momentum_rate, l2_decay, step_epochs, num_epochs):
    if lr_strategy == "piecewise_decay":
        return piecewise_decay(net, device_num, total_images, batch_size, lr,
                               momentum_rate, l2_decay, step_epochs)
    elif lr_strategy == "cosine_decay":
        return cosine_decay(net, device_num, total_images, batch_size, lr,
                            momentum_rate, l2_decay, num_epochs)
