from collections import Counter
import argparse

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from tactile_dataset import ShapeNetWhiskingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig):
    print("loading dataset ", DATA_DIR+"/"+SPLIT)
    dataset = ShapeNetWhiskingDataset(cfg=cfg, split=SPLIT, data_dir=DATA_DIR)
    print(dataset[0][0].shape, dataset[0][1])
    print(dataset.dataset_len)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=32,
        shuffle=True,
    )

    features, labels = None, None

    for batch in tqdm(dataloader):
        data, batch_labels = batch
        batch_features = data.view(data.size(0), -1).numpy()
        if features is None:
            features = batch_features
            labels = batch_labels
        else:
            features = np.concatenate((features, batch_features), axis=0)
            labels = np.concatenate((labels, batch_labels), axis=0)

    print("features", features.shape, "labels", labels.shape)

    rows_with_nan = np.sum(np.isnan(features).any(axis=1))
    print("Number of rows with NaN values:", rows_with_nan)
    rows_with_large_values = np.sum((features > 1e01).any(axis=1))
    print("Number of rows with values greater than 10:", rows_with_large_values)

    nans_per_row = np.isnan(features).sum(axis=1)
    max_nans = np.max(nans_per_row)
    print("Max number of NaNs in a row:", max_nans)

    avg_nans = np.mean(nans_per_row)
    print("Average number of NaNs per row:", avg_nans)

    max_value = np.nanmax(features)  # Ignore NaNs
    print("Max value in all rows:", max_value)

    features = np.nan_to_num(features, nan=0)

    NUM_CATEGORIES = 4
    label_counts = Counter(labels)
    # top4_label_counts = label_counts.most_common(NUM_CATEGORIES)
    # filter_labels = []
    # for label, count in top4_label_counts:
    #     filter_labels.append(label)
    #     print(f"Label {label}: {count} items")
    filter_labels = [49, 62, 100, 101]
    for label in filter_labels:
        print(f"Count for label {label}: {label_counts.get(label, 0)}")

    for k in range(2, NUM_CATEGORIES+1):
        mask = np.isin(labels, filter_labels[:k])
        features_k = features[mask]
        labels_k = labels[mask]

        split_idx = int(0.9 * len(features_k))
        X_train, X_test = features_k[:split_idx], features_k[split_idx:]
        y_train, y_test = labels_k[:split_idx], labels_k[split_idx:]

        print(f"Training {k}-category SVM...")
        svm = LinearSVC()
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)
        print("Evaluation Metrics:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM test")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/group_data/neuroagents_lab/training_datasets/tactile_whiskers/1000hz/processed",
        help="Directory for the dataset",
    )
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    SPLIT = args.split

    main()
