# Copyright 2026, Lawrence Livermore National Security, LLC and MADA contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The common package includes code commonly used across the entire
MADA project.

Modules:
    exceptions: Custom exceptions for MADA.
"""

from mada.common.exceptions import MADAUnsupportedDatabase

__all__ = ["MADAUnsupportedDatabase"]
