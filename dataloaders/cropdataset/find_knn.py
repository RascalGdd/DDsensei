from argparse import ArgumentParser
from pathlib import Path
import os
import faiss
import numpy as np
from cfg import *

if __name__ == '__main__':

    out = match_path
    file_src = fake_feature_path
    file_dst = real_feature_path
    k = 10


    features_ref = np.load(file_src)['crops'].astype(np.float32)
    features_ref = features_ref / np.sqrt(np.sum(np.square(features_ref), axis=1, keepdims=True))

    dim = features_ref.shape[1]
    print(f'Found {features_ref.shape[0]} crops for source dataset.')

    features_nn = np.load(file_dst)['crops'].astype(np.float32)
    features_nn = features_nn / np.sqrt(np.sum(np.square(features_nn), axis=1, keepdims=True))
    assert features_nn.shape[1] == dim
    print(f'Found {features_nn.shape[0]} crops for target dataset.')

    nn_index = faiss.IndexFlatL2(dim)
    nn_index.add(features_nn)

    D,I = nn_index.search(features_ref, k)

    np.savez_compressed(out, ind=I, dist=D)
    pass