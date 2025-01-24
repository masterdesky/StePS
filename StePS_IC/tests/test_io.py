#!/usr/bin/env python

import debugpy
import numpy as np

def attach_debugpy():
    debugpy.listen(("0.0.0.0", 5689))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached.")

def main():
    attach_debugpy()

    from stepsic.inputoutput import load_snapshot
    _, C, _, M = load_snapshot('examples/Glass_Nr224_Nhp32_D1860.hdf5')
    G = np.c_[C, np.zeros_like(C, dtype=np.float64), M]
    print(G.shape)

if __name__ == '__main__':
    main()