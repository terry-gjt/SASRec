import datetime
import os
import time
import argparse
import tensorflow as tf
#terry 20200528修改
# import tensorflow.compat.v1 as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    start = datetime.datetime.now()
    print("运行开始:%s"%start)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True)
    # parser.add_argument('--train_dir', required=True)
    #terry 20200528修改 使用默认数据集 原运行方式：python main.py --dataset=Video --train_dir=default
    # tf.disable_v2_behavior()
    parser.add_argument('--dataset', default="Video")
    parser.add_argument('--train_dir', default="default")

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):#判断文件目录是否存在，不存在就创建
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(list(vars(args).items()), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size #20200607 terry修改
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    config = tf.ConfigProto()
    #terry 20200528修改
    # config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    #terry 20200528修改
    # sess = tf.compat.v1.Session(config=config)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = Model(usernum, itemnum, args)
    sess.run(tf.initialize_all_variables())

    T = 0.0
    t0 = time.time()

    try:
        for epoch in range(1, args.num_epochs + 1):
        # for epoch in tqdm(list(range(1, 5)), total=num_batch, ncols=70, leave=False, unit='个'):#试了下，无法实现双进度条
        # for epoch in range(1, 2):

            for step in tqdm(list(range(num_batch)), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                         model.is_training: True})

            if epoch % 20 == 0:
                t1 = time.time() - t0
                T += t1
                print('Evaluating')
                t_test = evaluate(model, dataset, args, sess)
                t_valid = evaluate_valid(model, dataset, args, sess)
                print('')
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
    except Exception:
        sampler.close()
        f.close()
            # print(Exception.args)
        exit(1)

    f.close()
    sampler.close()
    end = datetime.datetime.now()
    print("运行完成: %s" % end)
    print('运行时间: %s 秒' % (end - start))

