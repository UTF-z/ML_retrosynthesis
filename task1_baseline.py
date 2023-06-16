import sys
sys.path.append('.')
from pathlib import Path
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from dataset.single_step import SingleStep
from tqdm import tqdm
from loguru import logger
from model.networks import Baseline
import argparse

def test(ckpt_path):
    data_dir = Path('resources')
    test_set = SingleStep(data_path=data_dir/'task1_test.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False, drop_last=False)
    model = Baseline.load_and_construct(ckpt_path)
    model.filter = FILTER
    model.topk = TOPK
    correct_num = 0
    total_num = 0
    for batch in tqdm(test_loader):
        c, t = model(batch, mode='val')
        correct_num += c
        total_num += t
    print(correct_num / total_num)

def inference(ckpt_path):
    data_dir = Path('resources')
    test_set = SingleStep(data_path=data_dir/'task1_test.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False, drop_last=False)
    model = Baseline.load_and_construct(ckpt_path)
    topks = []
    for batch in tqdm(test_loader):
        topk = model(batch, mode='infer')
        topks.append(topk)
    topks = torch.cat(topks, dim=0)
    print(topks.shape)
    

def train():
    now = time.strftime("%d-%H%M%S", time.localtime(time.time()))

    # get data
    data_dir = Path('resources')
    train_set = SingleStep(data_path=data_dir/'task1_train.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    val_set = SingleStep(data_path=data_dir/'task1_val.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    test_set = SingleStep(data_path=data_dir/'task1_test.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BS, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False, drop_last=False)

    # get model
    model_name = now + "_mlp.pkl"
    ckpt_dir = Path('checkpoint') / 'task1'
    ckpt_dir.mkdir(exist_ok=True)
    state_dict_path = ckpt_dir / model_name
    model_config = {
        'sizes': [2048, HIDDEN_SIZE, HIDDEN_SIZE, train_set.template_num],
        'dropout': DROPOUT,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'device': DEVICE,
        'topk': TOPK,
        'hash_table': list(train_set.temp_hash.keys()),
        'filter': FILTER,
        'num_worker': NUM_WORKERS,
        'multi_processing': True
    }
    model = Baseline(model_config)

    # log setting
    if FILTER:
        log_dir = f'log/task1_filter/{now}'
    else:
        log_dir = f'log/task1_baseline/{now}'
    os.makedirs(log_dir, exist_ok=True)
    log_dir = Path(log_dir)
    train_log_path = log_dir / f'train.log'
    val_log_path = log_dir / f'val.log'
    test_log_path = log_dir / f'test.log'
    try:
        os.remove(train_log_path)
    except Exception as e:
        pass
    try:
        os.remove(test_log_path)
    except Exception as e:
        pass
    try:
        os.remove(val_log_path)
    except Exception as e:
        pass
    logger.remove(None)
    logger.add(train_log_path, filter=lambda record: record['extra']['name'] == 'train')
    logger.add(val_log_path, filter=lambda record: record['extra']['name'] == 'val')
    logger.add(test_log_path, filter=lambda record: record['extra']['name'] == 'test')
    log_train = logger.bind(name='train')
    log_test = logger.bind(name='test')
    log_val = logger.bind(name='val')

    # train
    best_val_acc = 0
    with tqdm(range(EPOCH)) as train_bar:
        for i in range(EPOCH):
            loss_lst = []
            train_correct_cnt = 0
            train_total_cnt = 0
            for pack in train_loader:
                loss, correct_num, total_num, _ = model(pack, mode='train')
                # train log
                loss_lst.append(loss)
                train_correct_cnt += correct_num
                train_total_cnt += total_num
            avg_loss = np.mean(loss_lst)
            avg_acc = train_correct_cnt / train_total_cnt
            train_bar.set_postfix_str(f'loss: {avg_loss}, acc: {avg_acc}')
            log_train.info(f'loss: {np.mean(loss_lst)}, acc: {avg_acc}')
            train_bar.update()
            # validation
            if i % VAL_INT == 0:
                val_correct_cnt = 0
                val_total_cnt = 0
                for pack in val_loader:
                    correct_num, total_num = model(pack, mode='val')
                    val_correct_cnt += correct_num
                    val_total_cnt += total_num
                val_acc = val_correct_cnt / val_total_cnt
                if val_acc > best_val_acc:
                    model.save(state_dict_path)
                log_val.critical(f'epoch: {i}, val_acc: {val_acc}')

    log_train.critical('train finished') 
    model.load_ckpt(state_dict_path)
    test_correct_cnt = 0
    test_total_cnt = 0
    with tqdm(test_loader) as testbar:
        for pack in test_loader:
            correct_num, total_num = model(pack, mode='val')
            test_correct_cnt += correct_num
            test_total_cnt += total_num
            testbar.update()
    log_test.critical(f'test acc: {test_correct_cnt / test_total_cnt}')
    

if __name__ == "__main__":
    seed=8192
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", action='store_true', default=False)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_int", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--drop_out", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    EPOCH = args.epoch
    BS = args.batch_size
    VAL_INT = args.val_int
    DROPOUT = args.drop_out
    NUM_WORKERS = args.num_workers
    TOPK = args.topk
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    FILTER = args.filter
    HIDDEN_SIZE = args.hidden_size
    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = 'cpu'
    if args.mode == 'test':
        test('checkpoint/task1/15-202804_mlp.pkl')    
    elif args.mode == 'infer':
        inference('checkpoint/task1/15-202804_mlp.pkl')
    elif args.mode == 'train':
        train()



