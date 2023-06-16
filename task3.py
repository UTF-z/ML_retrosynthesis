import numpy as np
from pathlib import Path
import numpy 
from model.tree import Tree 
from model.tree import viz_search_tree
from model.networks import Baseline, MLP
from dataset.multi_step import MultiStep
from utils.utils import load_best
from tqdm import tqdm
from dataset.single_step import SingleStep
from utils.utils import get_reactants

def debug(model1_path, model2_path):
    task1_data_root = 'resources/task1_train.pkl'
    data1 = SingleStep(task1_data_root, 'resources/temp_hash.pkl')
    task3_data_root = 'resources/Multi-Step task'
    data3 = MultiStep(task3_data_root)
    # pack1 = data1[0]
    # model1 = Baseline.load_and_construct(model1_path)
    # model1.topk = 10
    # model1.multi_processing = False
    # model2 = MLP(sizes=[2048,512,128,1],dropout=0.1,act_func="elu",last_act="elu")
    # model2_state_dict = load_best(model2_path)
    # model2.load_state_dict(model2_state_dict)
    test_in = data3.make_test_fn()
    # expand_fn = model1.make_task1_infer_fn()
    # value_fn = model2.make_task2_infer_fn()
    # mol = pack1['smile']
    # temp = pack1['template']
    # reactants = get_reactants(mol, temp)
    # print(reactants)
    # tree = Tree(mol, test_in, expand_fn, model1.topk, value_fn, max_steps=500)
    # tree.fit()
    # pred_path = tree.transform()
    # print('pred_path', pred_path)
    fail_cnt = 0
    pack_cnt = 0
    in_cnt = 0
    total_len = 0
    with tqdm(data1) as tbar:
        for pack in data1:
            pack_cnt += 1
            mol = pack['smile']
            temp = pack['template']
            reactants = get_reactants(mol, temp)
            if reactants is None:
                fail_cnt += 1
            else:
                for reactant in reactants:
                    if test_in(reactant):
                        in_cnt += 1
                    total_len += 1
            tbar.set_postfix_str(f'fail_ratio: {(fail_cnt / pack_cnt):.3f}, in_ratio: {(in_cnt / total_len):.3f}')
            tbar.update()

    print(f'fail_ratio: {fail_cnt / total_len}')
    print(f'in_ratio: {in_cnt / total_len}')




def task3(model1_path, model2_path):
    task3_data_root = 'resources/Multi-Step task'
    data = MultiStep(task3_data_root)
    model1 = Baseline.load_and_construct(model1_path)
    model1.topk = 10
    model1.multi_processing = False
    model2 = MLP(sizes=[2048,512,128,1],dropout=0.1,act_func="elu",last_act="elu")
    model2_state_dict = load_best(model2_path)
    model2.load_state_dict(model2_state_dict)
    test_in = data.make_test_fn()
    expand_fn = model1.make_task1_infer_fn()
    value_fn = model2.make_task2_infer_fn()

    total_mol_cnt = len(data)
    success_cnt = 0
    length_ratio_lst = []
    with tqdm(data) as tbar:
        for mol, path in data:
            tree = Tree(mol, test_in, expand_fn, model1.topk, value_fn, max_steps=500)
            tree.fit()
            pred_path = tree.transform()
            if pred_path is not None:
                success_cnt += 1
                pred_length = len(pred_path[1]) 
                expert_length = len(path[1])
                length_ratio = pred_length / expert_length
                length_ratio_lst.append(length_ratio)
            tbar.set_postfix_str(f'success_cnt:{success_cnt}')
            tbar.update()
        # tree.viz_search_tree('tree_visual')
    success_rate = success_cnt / total_mol_cnt
    avg_length_ratio = np.mean(length_ratio_lst)
    print(success_rate, avg_length_ratio)

if __name__ == '__main__':
    # task3('checkpoint/task1/15-202804_mlp.pkl', 'checkpoint/task2/baseline/baseline_best.pth')
    debug('checkpoint/task1/15-202804_mlp.pkl', 'checkpoint/task2/baseline/baseline_best.pth')
    


