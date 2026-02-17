from nodes import *
from collections import deque

# Breadth-first Search
class BFS:
    def __init__(self, root:ChemNode, max_depth:int = 4):
        self.root = root
        self.max_depth = max_depth
        self.tree_built = False
        self.queue = deque([self.root])
        self.explored = set()

    def multi_step_planner(self):
        if self.tree_built:
            return
        
        while self.queue:
            node = self.queue.popleft()

            # skip if already explored
            if id(node) in self.explored:
                continue
            self.explored.add(id(node))

            for reaction in node.reactions:
                for precursor_smiles in reaction.precursors_smiles:
                    if precursor_smiles in node.ancestors: # if is loop
                        child = ChemNode(precursor_smiles, node.depth + 1, reaction, True)
                        reaction.precursors.append(child)
                    
                    elif node.depth + 1 < self.max_depth:
                        child = ChemNode(precursor_smiles, node.depth + 1, reaction)
                        child.ancestors.update(node.ancestors)
                        reaction.precursors.append(child)

                        # add unsolved to queue
                        if not child.solution:
                            self.queue.append(child)
                    else:
                        child = ChemNode(precursor_smiles, node.depth + 1, reaction, True)
                        reaction.precursors.append(child)

        self.backpropagate(self.root)
        self.tree_built = True
        return "Search is complete!"

    def backpropagate(self, node:ChemNode):
        """
        Backpropagate buyability up the tree
        """
        for reaction in node.reactions:
            for precursor in reaction.precursors:
                self.backpropagate(precursor)

        for reaction in node.reactions:
            reaction.is_solved = all([precursor.solution for precursor in reaction.precursors])

        if not node.solution:
            node.solution = any(reaction.is_solved for reaction in node.reactions)

    def get_progress(self):
        """Get current search progress"""
        return {
            'nodes_explored': len(self.explored),
            'nodes_in_queue': len(self.queue),
            'tree_built': self.tree_built
        }
