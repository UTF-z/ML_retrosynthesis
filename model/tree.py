import os
import numpy as np

class MolNode(object):
    def __init__(self, 
                 mol, 
                 init_value, 
                 parent=None, 
                 is_known=False,
                 zero_known_value=True) -> None:
        self.mol = mol                  # the molecule
        # value
        self.pred_value = init_value    # the v(mol)
        self.value = init_value         # 
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


class ReactionNode:
    def __init__(self, parent, cost, template):
        self.parent = parent        # The parent MolNode
        self.depth = self.parent.depth + 1
        self.cost = cost            # The cost of the reaction
        self.template = template
        self.children = []
        self.value = None           # [V(m | subtree_m) for m in children].sum() + cost
        self.succ_value = np.inf    # total cost for existing solution
        self.target_value = None    # V_target(self | whole tree)
        self.succ = None            # successfully found a valid synthesis route
        self.open = True            # before expansion: True, after expansion: False
        parent.children.append(self)

class Tree(object):
    def __init__(self, target_mol, known_mols, value_fn=None) -> None:
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn if not (value_fn is None) else self.value_fn
        self.root = self._add_mol_node(self.target_mol,None)
        self.succ = self.target_mol in self.known_mols

    def value_fn(self, fp):
        return 
    
    def _add_mol_node(self, mol, parent):
        is_known = mol in self.known_mols
        init_value = self.value_fn(mol)
        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known
        )
        return mol_node