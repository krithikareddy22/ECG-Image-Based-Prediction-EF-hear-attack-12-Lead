"""EchoNext locked test subset (owner, 2026-11-06: "save few in test for test also...split accordingly").

20 % of EchoNext patients, assigned by a fixed hash of patient_key (sha256 mod 5 == 0). The assignment depends on the
patient only, not on which rows have been downloaded, so it is the same in interim and final runs. Locked patients
are never used for training, early stopping, hyper-parameter or class-weight choice; they are only scored.
"""
import hashlib

import numpy as np


def locked(patient_keys):
    return np.array([int(hashlib.sha256(str(k).encode()).hexdigest(), 16) % 5 == 0 for k in patient_keys], dtype=bool)
