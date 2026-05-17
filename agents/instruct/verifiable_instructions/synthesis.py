#!/usr/bin/env python3
"""
Training instruction synthesis using IFEval training constraints.
"""

import random
import warnings
import logging
from typing import List, Dict

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

from . import instructions_registry
from . import instructions

class TrainingInstructionSynthesizer:
    """Instruction synthesis using training constraints."""
    
    def __init__(self):
        self.instruction_classes = instructions_registry.INSTRUCTION_DICT
        self.conflicts = instructions_registry.INSTRUCTION_CONFLICTS
        print(f"Loaded {len(self.instruction_classes)} training constraints")
    
    def are_instructions_compatible(self, instr1: str, instr2: str) -> bool:
        """Check if two instruction types are compatible."""
        if instr1 in self.conflicts and instr2 in self.conflicts[instr1]:
            return False
        if instr2 in self.conflicts and instr1 in self.conflicts[instr2]:
            return False
        return True
    
    def generate_single_instructions(self, num_per_type: int = 5) -> List[Dict]:
        """Generate dataset of single training instructions (one constraint each)."""
        dataset = []
        
        for instr_type, instr_class in self.instruction_classes.items():
            for i in range(num_per_type):
                try:
                    # Create instruction instance
                    instruction = instr_class(f"{instr_type}_{i}")

                    # Random parameters
                    description = instruction.build_description()
                    
                    # Extract kwargs from instruction instance
                    try:
                        kwargs = instruction.get_instruction_args()
                    except:
                        kwargs = {}
                    
                    dataset.append({
                        'instruction_id': [instr_type],
                        'kwargs': [kwargs],
                        'description': description
                    })
                except Exception as e:
                    # Some instructions may require specific parameters
                    continue
        
        return dataset
    
    def generate_compound_instructions(self, num_instructions: int = 100, max_constraints: int = 4) -> List[Dict]:
        """Generate dataset of compound training instructions (multiple constraints)."""
        dataset = []
        all_instruction_types = list(self.instruction_classes.keys())
        
        for i in range(num_instructions):
            # Random number of constraints (2 to max_constraints)
            num_constraints = random.randint(2, max_constraints)
            
            # Start with random base instruction
            base_instr = random.choice(all_instruction_types)
            selected_instructions = [base_instr]
            
            # Add compatible instructions
            remaining = [instr for instr in all_instruction_types if instr != base_instr]
            
            for _ in range(num_constraints - 1):
                compatible = []
                for candidate in remaining:
                    if all(self.are_instructions_compatible(candidate, selected) 
                           for selected in selected_instructions):
                        compatible.append(candidate)
                
                if not compatible:
                    break
                    
                next_instr = random.choice(compatible)
                selected_instructions.append(next_instr)
                remaining.remove(next_instr)
            
            # Create instruction instances and extract kwargs
            instruction_instances = []
            kwargs_list = []
            descriptions = []
            successful_instruction_ids = []
            
            for instr_type in selected_instructions:
                try:
                    instr_class = self.instruction_classes[instr_type]
                    instruction = instr_class(f"compound_{i}_{instr_type}")
                    description = instruction.build_description()
                    instruction_instances.append(instruction)
                    descriptions.append(description)
                    successful_instruction_ids.append(instr_type)
                    
                    # Extract kwargs
                    try:
                        kwargs = instruction.get_instruction_args()
                    except:
                        kwargs = {}
                    kwargs_list.append(kwargs)
                    
                except Exception:
                    continue
            
            if instruction_instances:  # Only add if we successfully created instructions
                dataset.append({
                    'instruction_id': successful_instruction_ids,
                    'kwargs': kwargs_list,
                    'descriptions': descriptions
                })
        
        return dataset
    
    def generate_full_dataset(self, single_per_type: int = 3, compound_count: int = 50) -> Dict:
        """Generate complete dataset with both single and compound training instructions."""
        print(f"Generating training dataset with {single_per_type} examples per constraint type...")
        single_instructions = self.generate_single_instructions(single_per_type)
        
        print(f"Generating {compound_count} compound training instructions...")
        compound_instructions = self.generate_compound_instructions(compound_count)
        
        dataset = {
            'single_instructions': single_instructions,
            'compound_instructions': compound_instructions,
            'total_single': len(single_instructions),
            'total_compound': len(compound_instructions),
            'total_instructions': len(single_instructions) + len(compound_instructions),
            'instruction_types': list(self.instruction_classes.keys())
        }
        
        print(f"Training dataset generated:")
        print(f"  Single instructions: {dataset['total_single']}")
        print(f"  Compound instructions: {dataset['total_compound']}")
        print(f"  Total instructions: {dataset['total_instructions']}")
        print(f"  Instruction types available: {len(dataset['instruction_types'])}")
        
        return dataset
