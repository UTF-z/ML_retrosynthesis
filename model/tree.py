import sys
sys.path.append('.')
import os
import numpy as np
from utils.utils import get_reactants
from tqdm import tqdm

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
        self.succ = is_known            # If the mol is successfully retrosynthesized.
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


class Tree(object):
    def __init__(self,
                 target_mol: str, 
                 test_in, 
                 expand_fn,
                 topk=10,
                 value_fn=None,
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
        self.tbar = tqdm(range(self.max_steps))

    def fit(self) -> None:
        while True:
            is_continue = self.one_step()
            if not is_continue:
                break
        return
    
    def one_step(self) -> bool:
        if self.steps > self.max_steps:
            # stop at the hard time limit!
            return False
        m = self.select()
        self.expand(m)
        self.steps += 1
        self.tbar.update()
        return self.update(m)
    
    def select(self):
        """
        Select a molecule from frontier.
        """
        minv = np.inf
        mid = 0
        for idx in range(len(self.frontier)):
            if self.frontier[idx].v_target() < minv:
                mid = idx
                minv = self.frontier[idx].v_target()
        m = self.frontier[mid]
        self.frontier.pop(mid)
        return m
    
    def expand(self, mol_node: MolNode):
        # calculate the expanding methods!
        templates, costs = self.expand_fn(mol_node.get_mol())
        reactant_lists = [get_reactants(self.target_mol,template) for template in templates]
        # build the subtree!
        for j in range(len(templates)):
            if (reactant_lists[j] is None):
                pass
            else:
                self._add_reaction_node(
                    parent=mol_node,
                    cost=costs[j],
                    template=templates[j],
                    reactant_list=reactant_lists[j]
                ) 
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
            self.frontier.append(mol_node)
        reaction_node.rn = cost
        for cld in reaction_node.children:
            reaction_node.rn += cld.rn
    
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
        # 2. check if the root!
        if start_node.parent is None:
            # stop searching!
            return False
        # 3. set the parent reaction!
        parent_reaction = start_node.parent
        parent_reaction.succ = False        
        # 4. check whether to percolate up!
        pr = parent_reaction.parent
        percolate_flag = True
        for ch in pr.children:
            if ch.succ is None or ch.succ:
                percolate_flag = False
                break
        # 5. return
        if percolate_flag:
            return self.pop_failure(pr)
        else:
            return True

    def pop_success(self,
                    start_node) -> bool:
        # 1. set the current node!
        start_node.succ = True
        # 2. check if the root!
        if start_node.parent is None:
            # stop searching!
            return False
        # 3. set the parent reaction!
        parent_reaction = start_node.parent   
        # 4. check whether to percolate up!
        pr = parent_reaction.parent
        percolate_flag = True
        for ch in parent_reaction.children:
            if not ch.succ:
                percolate_flag = False
                break
        # 5. return
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

    # def expand(self, mol_node, reactant_lists, costs, templates):
    #     assert not mol_node.is_known and not mol_node.children

    #     if costs is None:      # No expansion results
    #         assert mol_node.init_values(no_child=True) == np.inf
    #         if mol_node.parent:
    #             mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
    #         return self.succ

    #     assert mol_node.open
    #     ancestors = mol_node.get_ancestors()
    #     for i in range(len(costs)):
    #         self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
    #                                          mol_node, templates[i], ancestors)

    #     if len(mol_node.children) == 0:      # No valid expansion results
    #         assert mol_node.init_values(no_child=True) == np.inf
    #         if mol_node.parent:
    #             mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
    #         return self.succ

    #     v_delta = mol_node.init_values()
    #     if mol_node.parent:
    #         mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

    #     if not self.succ and self.root.succ:
    #         # logging.info('Synthesis route found!')
    #         self.succ = True

    #     return self.succ