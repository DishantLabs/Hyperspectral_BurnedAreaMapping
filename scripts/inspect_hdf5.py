"""Print the dataset tree of a Tanager HDF5 so you can set config.yaml paths."""
import sys
from src import data as D

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_hdf5.py <scene.h5>")
        raise SystemExit(1)
    D.print_tree(sys.argv[1])
