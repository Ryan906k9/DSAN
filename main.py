import paddle.nn.functional as F
import math
import argparse
import numpy as np
import paddle
from DSAN import DSAN
import data_loader


def load_data(root_path, src, tar, batch_size):
    # kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size)
    loader_tar = data_loader.load_training(root_path, tar, batch_size)
    loader_tar_test = data_loader.load_testing(root_path, tar, batch_size)
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    # LEARNING_RATE = args.lr / \
    #                 math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
    # print('learning rate{: .4f}'.format(LEARNING_RATE))

    # if args.bottleneck:
    #     optimizer = paddle.optim.SGD([
    #         {'params': model.feature_layers.parameters()},
    #         {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
    #         {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    #     ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.decay)
    # else:
    #     optimizer = paddle.optim.SGD([
    #         {'params': model.feature_layers.parameters()},
    #         {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    #     ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.decay)

    model.train()
    source_loader, target_train_loader, _ = dataloaders
    # for i,(a,b) in enumerate(source_loader()):
    #     print(i)
    #     print(b)
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        # print(iter_source.next())
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source, label_source
        data_target = data_target

        optimizer.clear_grad ( ) 
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, axis=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_lmmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with paddle.no_grad():
        for data, target in dataloader:
            data, target = data, target
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, axis=1), target).item()
            # print(pred)
            # pred = pred.max(1)[1]
            pred = paddle.argmax(pred, axis=1)
            # print(target)
            # print(pred)
            # correct += pred.equal(target.view_as(pred)).cpu().sum()
            # correct += paddle.sum(pred.equal(target)).numpy()[0]
            correct += paddle.sum(paddle.where(pred==paddle.squeeze(target), paddle.to_tensor(np.ones(pred.shape),dtype='int64'),
                                    paddle.to_tensor(np.zeros(pred.shape),dtype='int64'))).numpy()[0]
            # print(correct)

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='./work/data/Original_images')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='amazon/images/')
    parser.add_argument('--tar', type=str,
                         help='Target domain', default='webcam/images/')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    paddle.seed(SEED)

    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('start training ... ')

    dataloaders = load_data(args.root_path, args.src, args.tar, args.batch_size)
    model = DSAN(num_classes=args.nclass)

    correct = 0
    stop = 0
    # print(args.lr[0])
    if args.bottleneck:
        optimizer = paddle.optimizer.SGD(learning_rate=args.lr[0], parameters =
        model.parameters(),  weight_decay=args.decay)
    else:
        optimizer = paddle.optimizer.SGD(learning_rate=args.lr[0], parameters =
        model.parameters(), weight_decay=args.decay)
    # if args.bottleneck:
    #     optimizer = paddle.optimizer.SGD([
    #         {'params': model.feature_layers.parameters()},
    #         {'params': model.bottle.parameters(), 'lr': args.lr[1]},
    #         {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
    #     ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    # else:
    #     optimizer = paddle.optimizer.SGD([
    #         {'params': model.feature_layers.parameters()},
    #         {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
    #     ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    for epoch in range(1, args.nepoch + 1):
        stop += 1

        if args.bottleneck:
            optimizer.set_lr(args.lr[0] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75))
            optimizer.set_lr(args.lr[1] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75))
            optimizer.set_lr(args.lr[1] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75))
        else:
            optimizer.set_lr(args.lr[0] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75))
            optimizer.set_lr(args.lr[1] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75))
        train_epoch(epoch, model, dataloaders, optimizer)
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            # paddle.save(model, 'model.pkl')
            paddle.save(model.state_dict(), "model.pdparams")
            paddle.save(optimizer.state_dict(), "op.pdopt")
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break
