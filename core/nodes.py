"""
nodes.py

This module defines the `ChemNode` and `ReactNode` classes, which represent chemical compounds
and reactions in a retrosynthesis tree. These classes are used in Brute Force Search
to explore retrosynthesis pathways.
"""

from __future__ import annotations
import sqlite3
from utils import *
from typing import List, Set
import math

class ChemNode:
    """
    Basic class for a chemical compound node in the tree, holds all the information about the 
    compound (smiles, buyablity) as well as the tree structure (parent reaction, child reactions, depth).
    Also holds the static variables used by every ChemNode, such as the databases 
    (buyables, excluded, custom), the retrosim analyzer (for generating the child 
    reactions).
    """
    # Static variables used by every ChemNode
    buyables:sqlite3.Cursor = None # Cursor for buyable lookup
    excluded:sqlite3.Cursor = None # Excluded compounds
    analyzer:Retrosim = None # RDEnzyme + RetroBioCat single-step model
    rdenzyme_precursors:int = None # Max number of precursors to generate with RDEnzyme
    add_retrobiocat:bool = None # Include or not RetroBioCat suggestions

    def __init__(self, smiles:str, depth:int, parent_reaction:ReactNode,
                 forced_terminal:bool = False, max_precursors:int = None, add_retrobiocat:bool = None,
                 buyables:sqlite3.Cursor = None, custom:sqlite3.Cursor = None, excluded:sqlite3.Cursor = None, 
                 retrosim:Retrosim = None) -> None:
        """
        Initialize a ChemNode.

        :param smiles: SMILES string for the chemical compound.
        :param depth: Depth of the node in the tree.
        :param parent_reaction: Parent reaction node.
        :param buyables: Cursor for buyable database.
        :param virtual: Cursor for virtual database.
        :param excluded: Cursor for excluded database.
        :param retrosim: RDEnzyme analyzer.
        """
        # Chemical data
        self.smiles:str = smiles
        self.parent_reaction:ReactNode = parent_reaction
        self.ancestors:Set[str] = {smiles}
        self.depth:int = depth
        self.reactions:List[ReactNode] = []
        self.forced_terminal:bool = forced_terminal
        
        # Set up class variables -- these will be used by every ChemNode created
        if buyables is not None:
            ChemNode.buyables = buyables
        if excluded is not None:
            ChemNode.excluded = excluded
        if custom is not None:
            ChemNode.custom = custom
        if retrosim is not None:
            ChemNode.analyzer = retrosim
        if max_precursors is not None:
            ChemNode.rdenzyme_precursors = max_precursors
        if add_retrobiocat is not None:
            ChemNode.add_retrobiocat = add_retrobiocat
            
        self.availability:float = -1.0 # 0.0 = buyable, -1.0 = none
        # Check presence of the chemical in the buyable and virtual databases
        if not ChemNode.buyables or not ChemNode.custom or not ChemNode.excluded:
            print("Warning: No buyable or custom database provided.")
            return
        avail = check_available(smiles, ChemNode.buyables, ChemNode.excluded, ChemNode.custom)
        if avail == Availability.BUYABLE:
            self.availability = 0.0
        
        self.solution:bool = self.is_buyable()

        if (not self.solution) and (not self.forced_terminal): # If buyable, no need to generate reactions
            self.generate_reactions_rdenzyme() # Fills the reactions list
    
    def copy(self) -> ChemNode:
        """
        Copy this node (used in subtree generation)

        :return: A new ChemNode with the same properties as this one
        """
        new_node = ChemNode(self.smiles, self.depth, self.parent_reaction, True)
        new_node.reactions = []
        new_node.solution = self.solution
        new_node.availability = self.availability
        return new_node
    
    def generate_reactions_rdenzyme(self):
        """
        Populate the possible reactions for this node using RDEnzyme
        """
        # The single_step_retro function can output both RDEnzyme and RetroBioCat reactions
        results = ChemNode.analyzer.single_step_retro(target_molecule=self.smiles, max_precursors=ChemNode.rdenzyme_precursors, debug=False, retrobiocat=ChemNode.add_retrobiocat)
        for source, id, smarts, precursors, prob in results:
            self.reactions.append(ReactNode(reaction_name=str(id), smarts=smarts, source=source, parent_chemical=self.smiles, precursors_smiles=precursors.split('.')))
            
        if len(self.reactions) == 0:
            return
        
    def is_buyable(self) -> bool:
        """
        Check if this node is buyable 

        :return: True if the node is buyable, False otherwise
        """
        return self.get_availability() == Availability.BUYABLE 
    
    
    def get_availability(self) -> Availability:
        """
        Get the availability of this node (BUYABLE, NONE).
        Availability is stored as a float:
        - -1.0: Not available (NONE)
        -  0.0: Buyable (BUYABLE)

        :return: Availability enum value (BUYABLE, NONE)
        """
        
        if self.availability == -1.0:
            return Availability.NONE
        elif self.availability == 0.0:
            return Availability.BUYABLE
    
class ReactNode:
    """
    Class for a reaction node in the tree, holds all the information about the reaction (name, 
    smarts, source, name) as well as the tree structure (parent chemical, precursors).
    """
    def __init__(self, reaction_name:str, smarts:str, source:str, parent_chemical:ChemNode, precursors_smiles:list):
        """
        Initialize a ReactNode.

        :param reaction_name: Name/RHEA ID of the reaction.
        :param smarts: SMARTS string for the reaction.
        :param source: Source of the reaction (RetroBioCat or RHEA).
        :param parent_chemical: Parent ChemNode.
        """
        self.reaction_name = reaction_name
        self.smarts = smarts
        self.source = source 
        self.parent_chemical = parent_chemical
        self.precursors_smiles = precursors_smiles
        self.precursors:List['ChemNode'] = []  # Will hold ChemicalNodes representing precursors
        self.is_solved = False

    def copy(self) -> ReactNode:
        """
        Copy a ReactNode

        :return: A new ReactNode with the same properties as this one
        """
        react = ReactNode(self.reaction_name, self.smarts, self.source, self.parent_chemical, self.precursors_smiles)
        react.precursors = []
        return react

    def add_reactant(self, chemical_node:ChemNode):
        """
        Add a reactant to this reaction node

        :param chemical_node: The chemical node to add as a reactant
        """
        self.precursors.append(chemical_node)    
    
        
    