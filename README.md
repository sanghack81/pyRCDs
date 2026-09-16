# pyRCDs: Relational Causal Model and Tools

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![Tests](https://github.com/sanghack81/pyRCDs/actions/workflows/pytest.yml/badge.svg)](https://github.com/sanghack81/pyRCDs/actions)
[![codecov](https://codecov.io/gh/sanghack81/pyRCDs/branch/master/graph/badge.svg)](https://codecov.io/gh/sanghack81/pyRCDs)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**pyRCDs** implements relational causal models (RCMs) [[1](#1),[2](#2),[3](#3),[4](#4)], including relational schemas, skeletons, and ground graphs.

It also includes utilities for path semantics and theoretical evaluation. For structure learning from data, see [`RRCD`](https://github.com/sanghack81/RRCD) [[5](#5)].

## Installation

We recommend using Python 3.12 or newer. Since visual abstractions rely on `pygraphviz`, you need the `graphviz` C-library installed natively on your system.

**1. Install system dependencies (macOS):**
```bash
brew install graphviz
```

**2. Install the package:**
```bash
# Clone the repository and install dependencies
git clone https://github.com/sanghack81/pyRCDs.git
cd pyRCDs
pip install .

# For development and testing environments:
pip install .[dev,test]
```

## Example

```python
from pyrcds.domain import EntityClass, RelationalSchema

person = EntityClass("Person", ["age"])
schema = RelationalSchema({person}, set())
print(schema)  # RSchema(Person)
```

## Features

- **Relational Domain Representations**: Classes representing Entity Classes, Relationship Classes, and full Relational Schemas.
- **Relational Skeleton Formulation**: Graph-level instantiation models from schemas supporting intricate graph traversal.
- **Syntactic Operations & Paths**: Complex algorithms to ascertain ground graph dependencies, robust orienting functions, relational variables, and conditional independence testers.
- **Structure Visualization**: Seamless drawing capabilities generating aesthetic graph projections (via `networkx` and `pygraphviz`).

---

## References

<a id="1">[1]</a> 
Marc Maier, Brian Taylor, Huseyin Oktay, and David Jensen (2010). AAAI. Learning Causal Models of Relational Domains.   

<a id="2">[2]</a>
Marc Maier, Katerina Marazopoulou, David Arbour, and David Jensen (2013). UAI. A Sound and Complete Algorithm for Learning Causal Models from Relational Data 

<a id="3">[3]</a>
Sanghack Lee and Vasant Honavar (2016). AAAI. On Learning Causal Models from Relational Data

<a id="4">[4]</a>
Sanghack Lee and Vasant Honavar (2016). UAI. A Characterization of Markov Equivalence Classes for Relational Causal Model with Path Semantics. 

<a id="5">[5]</a>
Sanghack Lee and Vasant Honavar (2019). UAI. Towards Robust Relational Causal Discovery.
