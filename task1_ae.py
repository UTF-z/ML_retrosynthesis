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
from model.networks import AEClassifier

if __name__ == '__main__':
    seed=8192
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    EPOCH = 10000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VAL_INT = 10
    DROPOUT = 0.5
    now = time.strftime("%d-%H%M%S", time.localtime(time.time()))

    # get data
    data_dir = Path('resources')
    train_set = SingleStep(data_path=data_dir/'task1_train.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    val_set = SingleStep(data_path=data_dir/'task1_val.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    test_set = SingleStep(data_path=data_dir/'task1_test.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, drop_last=False)

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

    # get model
    model_name = now + "_mlp.pkl"
    ckpt_dir = Path('checkpoint') / 'task1'
    ckpt_dir.mkdir(exist_ok=True)
    state_dict_path = ckpt_dir / model_name
    model_config = {
        'header_size': [512, 512, train_set.template_num],
        'dropout': DROPOUT,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'device': DEVICE,
        'topk': 10,
        'hash_table': list(train_set.temp_hash.keys()),
        'filter': FILTER,
        'num_worker': NUM_WORKERS,
        'multi_processing': True
    }
    model = AEClassifier(model_config).to(DEVICE)
    # train
    best_val_acc = 0
    with tqdm(range(EPOCH)) as train_bar:
        for i in range(EPOCH):
            loss_lst = []
            train_correct_cnt = 0
            train_total_cnt = 0
            for pack in train_loader:
                loss, correct_num, total_num, loss_info = model(pack, mode='train')
                # train log
                loss_lst.append(loss)
                train_correct_cnt += correct_num
                train_total_cnt += total_num
            avg_loss = np.mean(loss_lst)
            avg_acc = train_correct_cnt / train_total_cnt
            train_bar.set_postfix_str(f'loss: {np.mean(loss_lst)}, acc: {avg_acc}')
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
    model.load(state_dict_path)
    test_correct_cnt = 0
    test_total_cnt = 0
    with tqdm(test_loader) as testbar:
        for pack in test_loader:
            correct_num, total_num = model(pack, mode='val')
            test_correct_cnt += correct_num
            test_total_cnt += total_num
            testbar.update()
    log_test.critical(f'test acc: {test_correct_cnt / test_total_cnt}')