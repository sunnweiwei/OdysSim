# Copyright 2025 Individual Contributor: OdysSim Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verifiable Instructions - Lightweight verifiable instructions for training.

This package provides instruction classes and synthesis tools for generating
instruction following datasets using verifiable training constraints.
"""

from .instructions_registry import INSTRUCTION_CONFLICTS, INSTRUCTION_DICT
from .synthesis import TrainingInstructionSynthesizer

__version__ = "0.1.0"
__all__ = ["TrainingInstructionSynthesizer", "INSTRUCTION_DICT", "INSTRUCTION_CONFLICTS"]
