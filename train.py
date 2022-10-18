import os
import argparse
import time
import random
import numpy as np
import paddle
from paddle.distributed import ParallelEnv

from optimizer import create_optimizer
from dataset import ImageNetDataset
from utils import print_arguments

from models import MobileOne
from models.mobilenetv3 import MobileNetV3_large_x1_0


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    # yapf: disable
    parser.add_argument('--batch_size', type=int, default=128, help="Single Card Minibatch size.")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Whether to use GPU or not.")
    parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate used to fine-tune pruned model.")
    parser.add_argument('--lr_strategy', type=str, default="piecewise_decay", help="The learning rate decay strategy.")
    parser.add_argument('--l2_decay', type=float, default=3e-5, help="The l2_decay parameter.")
    parser.add_argument('--ls_epsilon', type=float, default=0.0, help="Label smooth epsilon.")
    parser.add_argument('--momentum_rate', type=float, default=0.9, help="The value of momentum_rate.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of total epochs.")
    parser.add_argument('--resume', type=bool, default=False, help="Resume training.")
    parser.add_argument('--pretrained_model', type=str, default="./output_models", help="model save directory.")
    parser.add_argument('--log_period', type=int, default=10, help="Log period in batches.")
    parser.add_argument('--model_save_dir', type=str, default="./output_models", help="model save directory.")
    parser.add_argument('--step_epochs', nargs='+', type=int, default=[10, 20, 30], help="piecewise decay step")
    # yapf: enable
    return parser


def load_dygraph_pretrain(model, path=None, load_static_weights=False):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def main():
    reader_num_workers = 4
    shuffle = True
    seed = 111
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = ImageNetDataset(
        data_dir='/paddle/dataset/ILSVRC2012/', mode='train')
    val_dataset = ImageNetDataset(
        data_dir='/paddle/dataset/ILSVRC2012/', mode='val')
    total_images = len(train_dataset)
    print('total train images:', total_images)
    class_dim = 1000

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1

    place = paddle.set_device('gpu' if FLAGS.use_gpu else 'cpu')
    # model definition
    if use_data_parallel:
        paddle.distributed.init_parallel_env()

    # net = MobileOne(num_classes=class_dim)
    net = MobileNetV3_large_x1_0(num_classes=class_dim)
    if FLAGS.resume:
        load_dygraph_pretrain(net, FLAGS.pretrained_model, True)

    print("Model summary:")
    paddle.summary(net, (1, 3, 224, 224))

    opt, lr = create_optimizer(net, FLAGS.lr_strategy, trainer_num,
                               total_images, FLAGS.batch_size, FLAGS.lr,
                               FLAGS.momentum_rate, FLAGS.l2_decay,
                               FLAGS.step_epochs, FLAGS.num_epochs)

    if use_data_parallel:
        net = paddle.DataParallel(net)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=shuffle,
        drop_last=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        places=place,
        return_list=True,
        num_workers=reader_num_workers)

    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        drop_last=False,
        return_list=True,
        num_workers=reader_num_workers)

    @paddle.no_grad()
    def test(epoch, net):
        net.eval()
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []

        eval_reader_cost = 0.0
        eval_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for data in valid_loader():
            eval_reader_cost += time.time() - reader_start
            image = data[0]
            label = data[1]

            eval_start = time.time()

            out = net(image)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            eval_run_cost += time.time() - eval_start
            batch_size = image.shape[0]
            total_samples += batch_size

            if batch_id % FLAGS.log_period == 0:
                log_period = 1 if batch_id == 0 else FLAGS.log_period
                print(
                    "Eval epoch[{}] batch[{}] - top1: {:.6f}; top5: {:.6f}; avg_reader_cost: {:.6f} s, avg_batch_cost: {:.6f} s, avg_samples: {}, avg_ips: {:.3f} images/s".
                    format(epoch, batch_id,
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), eval_reader_cost /
                           log_period, (eval_reader_cost + eval_run_cost) /
                           log_period, total_samples / log_period,
                           total_samples / (eval_reader_cost + eval_run_cost)))
                eval_reader_cost = 0.0
                eval_run_cost = 0.0
                total_samples = 0
            acc_top1_ns.append(np.mean(acc_top1.numpy()))
            acc_top5_ns.append(np.mean(acc_top5.numpy()))
            batch_id += 1
            reader_start = time.time()

        print("Final eval epoch[{}] - acc_top1: {:.6f}; acc_top5: {:.6f}".
              format(epoch,
                     np.mean(np.array(acc_top1_ns)),
                     np.mean(np.array(acc_top5_ns))))
        return np.mean(np.array(acc_top1_ns))

    def cross_entropy(input, target, ls_epsilon):
        if ls_epsilon > 0:
            if target.shape[-1] != class_dim:
                target = paddle.nn.functional.one_hot(target, class_dim)
            target = paddle.nn.functional.label_smooth(
                target, epsilon=ls_epsilon)
            target = paddle.reshape(target, shape=[-1, class_dim])
            input = -paddle.nn.functional.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = paddle.nn.functional.cross_entropy(
                input=input, label=target)
        avg_cost = paddle.mean(cost)
        return avg_cost

    def train(epoch, net):

        net.train()
        batch_id = 0

        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for data in train_loader():
            train_reader_cost += time.time() - reader_start

            image = data[0]
            label = data[1]

            train_start = time.time()
            out = net(image)
            avg_cost = cross_entropy(out, label, FLAGS.ls_epsilon)

            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
            avg_cost.backward()
            opt.step()
            opt.clear_grad()
            lr.step()

            loss_n = np.mean(avg_cost.numpy())
            acc_top1_n = np.mean(acc_top1.numpy())
            acc_top5_n = np.mean(acc_top5.numpy())

            train_run_cost += time.time() - train_start
            batch_size = image.shape[0]
            total_samples += batch_size

            if batch_id % FLAGS.log_period == 0:
                log_period = 1 if batch_id == 0 else FLAGS.log_period
                print(
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {:.6f}; top1: {:.6f}; top5: {:.6f}; avg_reader_cost: {:.6f} s, avg_batch_cost: {:.6f} s, avg_samples: {}, avg_ips: {:.3f} images/s".
                    format(epoch, batch_id,
                           lr.get_lr(), loss_n, acc_top1_n, acc_top5_n,
                           train_reader_cost / log_period,
                           (train_reader_cost + train_run_cost) / log_period,
                           total_samples / log_period, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            batch_id += 1
            reader_start = time.time()

    # train loop
    best_acc1 = 0.0
    best_epoch = 0
    for i in range(FLAGS.num_epochs):
        train(i, net)
        acc1 = test(i, net)
        if paddle.distributed.get_rank() == 0:
            model_prefix = os.path.join(FLAGS.model_save_dir,
                                        "epoch_" + str(i))
            paddle.save(net.state_dict(), model_prefix + ".pdparams")
            paddle.save(opt.state_dict(), model_prefix + ".pdopt")

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = i
            if paddle.distributed.get_rank() == 0:
                model_prefix = os.path.join(FLAGS.model_save_dir, "best_model")
                paddle.save(net.state_dict(), model_prefix + ".pdparams")
                paddle.save(opt.state_dict(), model_prefix + ".pdopt")

    # save model
    if paddle.distributed.get_rank() == 0:
        # load best model
        load_dygraph_pretrain(net,
                              os.path.join(FLAGS.model_save_dir, "best_model"))

        path = os.path.join(FLAGS.model_save_dir, "inference_model", 'model')
        paddle.jit.save(
            net,
            path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 3, 224, 224], dtype='float32')
            ])


if __name__ == '__main__':
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    main()
