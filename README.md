# some stats functions i wrote

Very basic repo containing some statistical functions written to compensate where I found scipy lacking.

Currently contains two modules:
- OLSE, for very basic linear statistics
- LMoments, for fitting distributions efficiently

### Guide

A practical description of the included modules is included in the `example-nb` ipython notebook.

### Installation
Clone this repo, create a new venv (optional-ish), and from the base directory run

```
pip install -e ./
```

This will install the package. To then use in python add

```
from mystatsfunctions import OLSE,LMoments
```

The individual classes or functions are then accessed by eg. `OLSE.simple()`
