"""lithos_repair — DRC repair primitives and fix-graph engine.

Given a list of violations from ``lithos-drc`` and the rule DB from
``lithos-core``, this package applies fixes. It reads each violated rule's
``fix_metadata`` (allowed/forbidden action classes), evaluates the
constraint AST against the local geometry, and chooses a remedy. The
``rule_relation`` cross-reference graph drives anticipating downstream
violations.

Empty for now — package is scaffolded so the workspace builds.
"""
