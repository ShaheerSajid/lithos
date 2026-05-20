"""lithos_core — typed IR, rule DB, and PDK metadata.

The bottom of the lithos stack. Every other package depends on this one.

Public surface:

* :mod:`lithos_core.ir`        — the constraint IR (`Constraint`, `CheckExpr`,
                                  `LayerExpr`, conditions).
* :mod:`lithos_core.fix`       — the fix-metadata schema (`FixMetadata`).
* :mod:`lithos_core.db`        — SQLite-backed rule DB (`RuleDB`, `Rule`).
* :mod:`lithos_core.metadata`  — PDK metadata YAML loader (`PDKMetadata`).
"""

from lithos_core.ir import (
    Constraint,
    ConstraintBranch,
    LayerRef,
    LayerBool,
    LayerSize,
    LayerSelect,
    LayerEdges,
    LayerHoles,
    LayerConnect,
    WidthCheck,
    SpacingCheck,
    EnclosureCheck,
    AreaCheck,
    DensityCheck,
    AntennaCheck,
    ExistenceCheck,
    ParallelRunLength,
    WidthBand,
    LengthBand,
    EdgeOrientation,
    LayerPresence,
)
from lithos_core.fix import FixBranch, FixMetadata
from lithos_core.db import Rule, RuleDB
from lithos_core.metadata import PDKMetadata, load_metadata
from lithos_core.categories import CategoryConfig, CategoryDef, load_categories
from lithos_core.layers import LayerDef, LayersFile, load_layers_file

__all__ = [
    "Constraint",
    "ConstraintBranch",
    "LayerRef",
    "LayerBool",
    "LayerSize",
    "LayerSelect",
    "LayerEdges",
    "LayerHoles",
    "LayerConnect",
    "WidthCheck",
    "SpacingCheck",
    "EnclosureCheck",
    "AreaCheck",
    "DensityCheck",
    "AntennaCheck",
    "ExistenceCheck",
    "ParallelRunLength",
    "WidthBand",
    "LengthBand",
    "EdgeOrientation",
    "LayerPresence",
    "FixBranch",
    "FixMetadata",
    "Rule",
    "RuleDB",
    "PDKMetadata",
    "load_metadata",
    "CategoryConfig",
    "CategoryDef",
    "load_categories",
    "LayerDef",
    "LayersFile",
    "load_layers_file",
]
