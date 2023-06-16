import sys
sys.path.append('.')
import os
from collections import defaultdict
import numpy as np
from utils.utils import get_reactants
from tqdm import tqdm
import networkx as nx
from graphviz import Digraph
from queue import Queue


MAX_STEPS = 500

class MolNode(object):
    def __init__(self, 
                 mol: str, 
                 init_value, 
                 parent=None, 
                 is_known=False,
                 zero_known_value=True) -> None:
        self.mol = mol                  # the molecule
        # value
        self.pred_value = init_value    # the v(mol)
        self.rn = init_value            # 
        self.succ_value = np.inf        # total cost for existing solution
        # State
        self.is_known = is_known        # If the mol is known.
        self.succ = is_known if is_known else None    # If the mol is successfully retrosynthesized.
        # The pointers
        self.depth = 0 if parent is None else parent.depth # the depth of this node. (reaction and mol view as the same one.)
        self.children = []              # All the children reactions.
        self.parent = parent            # the parent reaction node
        if parent is not None:
            parent.children.append(self)
        # The state
        self.open = True                # before expansion: True, after expansion: False

    def v_self(self):
        """
        :return: V_self(self | subtree)
        """
        return self.rn

    def v_target(self):
        """
        :return: V_target(self | whole tree)
        """
        return self.rn if (self.parent is None) else self.parent.v_target()
    
    def get_mol(self):
        return self.mol
    
    def get_ancestors(self):
        if self.parent is None:
            return {self.mol}

        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors
    
    def has_children(self):
        return (len(self.children) > 0)
    
    def serialize(self):
        text = '%s' % (self.mol)
        return text


class ReactionNode(object):
    def __init__(self, parent, cost, template):
        self.parent = parent        # The parent MolNode
        self.depth = self.parent.depth + 1
        self.cost = cost            # The cost of the reaction
        self.template = template    # The template for the action
        self.children = []          # 
        self.rn = None           # sum([V(m | subtree_m) for m in children]) + cost
        self.succ_value = np.inf    # total cost for existing solution
        self.target_value = None    # V_target(self | whole tree)
        self.succ = None            # successfully found a valid synthesis route
        self.open = True            # before expansion: True, after expansion: False
        parent.children.append(self)

    def v_self(self):
        """
        :return: V_self(self | subtree)
        """
        return self.rn

    def v_target(self):
        """
        :return: V_target(self | whole tree)
        """
        return self.target_value

    def gen_rn(self):
        self.rn = self.cost
        for cld in self.children:
            self.rn += cld.value

    def serialize(self):
        return (self.template).replace(':', '_')
    
    def has_children(self):
        return True


class Tree(object):
    def __init__(self,
                 target_mol: str, 
                 test_in, 
                 expand_fn,
                 topk=10,
                 value_fn=None,
                 template_hash=None,
                 max_steps=MAX_STEPS) -> None:
        self.target_mol = target_mol
        self.test_in = test_in
        self.value_fn = value_fn if not (value_fn is None) else self.value_fn
        self.expand_fn = expand_fn
        self.root = self._add_mol_node(self.target_mol,None)
        self.frontier = [self.root]
        self.steps = 0                          # The max step shall be kept under 500
        self.max_steps = max_steps
        self.topk = topk
        self.succ = self.test_in(self.target_mol)
        self.failed = set()
        self.node_size = 0
        self.nodes = {}
        self.success_nodes = {}
        self.failure_nodes = {}

    def fit(self,viz_dir=None) -> None:
        lastid = 0
        cnt = 0
        os.makedirs(viz_dir, exist_ok=True)
        for file_name in os.listdir(viz_dir):
            curid = int(file_name.split("_")[1]) \
                  if (len(file_name.split("_")) > 1) else 0
            lastid = curid if lastid < curid else lastid
        while True:
            cnt += 1
            is_continue = self.one_step()
            if False:
                self.viz(viz_dir,lastid+1,cnt)
            if not is_continue:
                break
        self.viz(viz_dir,lastid+1,cnt)
        return
    
    def one_step(self) -> bool:
        if self.steps > self.max_steps:
            # stop at the hard time limit!
            return False
        self.steps += 1
        m = self.select()
        self.expand(m)
        return self.update(m)
    
    def select(self):
        """
        Select a molecule from frontier.
        """
        minv = np.inf
        mid = 0
        for idx in range(len(self.frontier)):
            if self.frontier[idx].mol in self.failed:
                continue
            if self.frontier[idx].v_target() < minv:
                mid = idx
                minv = self.frontier[idx].v_target()
        assert len(self.frontier) > 0
        m = self.frontier[mid]
        self.frontier.pop(mid)
        return m
    
    def get_all_parents(self, node_x):
        ans = [node_x.mol]
        while node_x.parent is not None:
            if isinstance(node_x.parent,MolNode):
                ans.append(node_x.parent.mol)
            node_x = node_x.parent
        return ans

    def expand(self, mol_node: MolNode):
        # calculate the expanding methods!
        templates, costs = self.expand_fn(mol_node.get_mol())
        reactant_lists = [get_reactants(self.target_mol,template) for template in templates]
        # build the subtree!
        all_parents = self.get_all_parents(mol_node)
        for j in range(len(templates)):
            if (reactant_lists[j] is None):
                continue
            if (np.array([m in all_parents for m in reactant_lists[j]],dtype=bool).any()):
                continue
            self.node_size += 1 if self._add_reaction_node(
                parent=mol_node,
                cost=costs[j],
                template=templates[j],
                reactant_list=reactant_lists[j]
            ) else 0            
        # print("node: ",self.node_size)
        # print("frontier",len(self.frontier))
        return
        
    def _add_mol_node(self, mol, parent):
        is_known = self.test_in(mol)
        init_value = self.value_fn(mol)
        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known
        )
        return mol_node
    
    def _add_reaction_node(self, 
                           parent, 
                           cost, 
                           template, 
                           reactant_list):
        reaction_node = ReactionNode(parent=parent,
                                     cost=cost,
                                     template=template)
        for reactant in reactant_list:
            mol_node = self._add_mol_node(reactant,reaction_node)
            if reactant in self.success_nodes:
                mol_node.succ = True
            elif reactant in self.failure_nodes:
                mol_node.succ = False
                # fail
                reaction_node.parent = None
                parent.children.remove(reaction_node)
                self.remove_frontier(mol_node,succ=False)
                return False
            else:
                if mol_node.succ is None or not mol_node.succ:
                    self.frontier.append(mol_node)
        reaction_node.rn = cost
        for cld in reaction_node.children:
            reaction_node.rn += cld.rn
        return True
    
    def transform(self):
        if not self.succ:
            return None
        else:
            success_route = (self.target_mol,[])
            self.traverse(self.root,
                          success_route)
            return success_route

    def traverse(self,
                 root: MolNode,
                 success_route: tuple):
        # 1. erroneous external calls
        if not self.succ:
            return
        # 2. a leaf
        if not root.has_children():
            return
        # 3. get the succeeded child
        succeeded_reaction = None
        for ch in root.children:
            if not (ch.succ is None) and ch.succ:
                succeeded_reaction = ch
                success_route[1].append(succeeded_reaction.template)
                break
        # 4. traverse through children
        for grandchild in succeeded_reaction.children:
            self.traverse(grandchild,success_route)        
        return

    def update(self, start_node: MolNode) -> bool:
        continue_flag = True
        # check failure!
        if (not start_node.has_children()):
            if not start_node.succ:
                continue_flag = self.pop_failure(start_node)
            else:
                continue_flag = self.pop_success(start_node)
        # check suceess!
        for ch in start_node.children:
            is_leaf = True
            for grandchildren in ch.children:
                if not grandchildren.succ:
                    is_leaf = False
                    break
            if is_leaf:
                ch.succ = True
                continue_flag = self.pop_success(start_node)
                break
        self.update_V(start_node)
        return continue_flag
    
    def pop_failure(self,
                    start_node: MolNode) -> bool:
        # 1. set the current node!
        start_node.succ = False
        self.failure_nodes[start_node.mol] = 1
        self.failed.add(start_node.mol)
        # 2. check if the root!
        if start_node.parent is None:
            # stop searching!
            return False
        # 3. set the parent reaction!
        parent_reaction = start_node.parent
        parent_reaction.succ = False       
        # 4. remove_frontier recursively
        self.remove_frontier(start_node, succ=False)
        # 5. check whether to percolate up!
        pr = parent_reaction.parent
        percolate_flag = True
        for ch in pr.children:
            if ch.succ is None or ch.succ:
                percolate_flag = False
                break
        # 6. return
        if percolate_flag:
            return self.pop_failure(pr)
        else:
            return True
        
    def remove_frontier(self, start_node: MolNode, succ: bool):
        """
                r1
            /   |   \
          /     |     \
        m1  star_node  m3
            /   |   \
            r2  r3  r4            
        """
        # 1. remove all its siblings from frontier
        if not succ:
            parent_reaction = start_node.parent
            [self.frontier.remove(sibling) 
            if sibling in self.frontier else None 
            for sibling in parent_reaction.children] 
        # 2. remove all its children recursively
        [[self.remove_frontier(grandchild, succ=False) 
          for grandchild in ch.children 
          if ch.succ is None] # both succeeded nodes and failed nodes have been handled before. 
        for ch in start_node.children]
        return

    def pop_success(self,
                    start_node) -> bool:
        # 1. set the current node!
        start_node.succ = True
        self.success_nodes[start_node.mol] = 1
        # 2. check if the root!
        if start_node.parent is None:
            # stop searching!
            return False
        # 3. set the parent reaction!
        parent_reaction = start_node.parent   
        # 4. remove_frontier recursively
        self.remove_frontier(start_node, succ=True)
        # 5. check whether to percolate up!
        pr = parent_reaction.parent
        percolate_flag = True
        for ch in parent_reaction.children:
            if not ch.succ:
                percolate_flag = False
                break
        # 6. return
        if percolate_flag:
            parent_reaction.succ = True     
            return self.pop_success(pr)
        else:
            return True
    
    def update_V(self, start_node: MolNode):
        ptr = start_node
        old_rn = ptr.rn
        new_rn = np.inf
        for cld in ptr.children:
            new_rn = min(new_rn, cld.rn)
        delta = new_rn - old_rn
        ptr.rn += delta
        while ptr != self.root:
            parent = ptr.parent
            parent.rn += delta
            parent.target_value += delta 
            if parent.rn >= parent.parent.rn:
                break
            ptr = parent.parent
            delta = parent.rn - ptr.rn
            ptr.rn += delta
        incre = start_node.v_target() - start_node.rn
        for cld in start_node.children:
            cld.target_value = cld.rn + incre

    def value_fn(self, fp):
        return 0.
    
    def viz_search_tree(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = '%.3f' % node.cost
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()


    # def expand(self, mol_node, reactant_lists, costs, templates):
    #     assert not mol_node.is_known and not mol_node.children

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()
            if node.has_children():
                color = 'lightgrey'
            else:
                color = 'aquamarine'
            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'
            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'
            elif node.succ is not None and not node.succ:
                color = 'red'
            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = '%.3f' % node.cost
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)
            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()

    def viz(self, 
            viz_dir: str,
            mol_id: int,
            cur_step: int):
        if not (viz_dir is None):
            os.makedirs(viz_dir, exist_ok=True)
            f = '%s/mol_%d_%d_search_tree' % (viz_dir, mol_id, cur_step)
            self.viz_search_tree(f)