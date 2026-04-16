#!/usr/bin/env python3
"""Print a summary of all entity types in a DXF file."""

import sys
from collections import Counter
import ezdxf

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} input.dxf")
    sys.exit(1)

doc = ezdxf.readfile(sys.argv[1])
msp = doc.modelspace()

counts = Counter(e.dxftype() for e in msp)

print("Entity types found:")
for etype, count in counts.most_common():
    print(f"  {etype}: {count}")