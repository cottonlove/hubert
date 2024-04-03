from pathlib import Path
import logging
import argparse

import torch
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cluster(args): #cluster에 대한 것을 .pt로 저장해둔 것
    with open(args.subset) as file:
        subset = [line.strip() for line in file]

    logger.info(f"Loading features from {args.in_dir}")
    features = []
    for path in subset:
        in_path = args.in_dir / path
        features.append(np.load(in_path.with_suffix(".npy"))) #encode.py에서 discrte/soft units으로 뽑아(self.kmeans.predict) .npy 파일에 저장한 것들
    features = np.concatenate(features, axis=0)

    logger.info(f"Clustering features of shape: {features.shape}")
    kmeans = KMeans(n_clusters=args.n_clusters).fit(features) #return Fitted estimator.
    # 근데 이게 cluster를 만들라면, 첨에는 ssl features에서 fit을 시켜야하는데 
    # encode.py에서는 load하는 모델이 SoftVC에서 fine-tuning한 HuBERT-Soft or HuBERT-Discrete임
    # 그래서 .npy로 features 불러오러면 그냥 pretrained HuBERT로 ssl features을 얻고 걜 저장한 후 cluster.py를 실행해서 저장해야할 듯?
    # encode.py에서는 soft unit encoder 학습을 위해 필요한 discrete units을 뽑기 위해 HuBERT-Discrete을 로드함
    # 내가 새로 cluster fit하고 싶으면 (LibriSpeech)로, 내가 짠 model.py의 HubertSSL로 ssl features 뽑아서 .npy로 저장해두면 될 듯

    checkpoint_path = args.checkpoint_dir / f"kmeans_{args.n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        checkpoint_path,
        {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster speech features features.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the encoded dataset",
        type=Path,
    )
    parser.add_argument(
        "subset",
        matavar="subset",
        help="path to the .txt file containing the list of files to cluster",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory",
        type=Path,
    )
    parser.add_argument(
        "--n-clusters",
        help="number of clusters",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    cluster(args)
