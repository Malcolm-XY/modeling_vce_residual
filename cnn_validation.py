# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 00:29:44 2024

@author: 18307
"""

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset #, Subset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# %% CNN
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_model(model, train_loader, device, optimizer, criterion, epochs=30):
    """
    Train a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        device (torch.device): Device to use for training.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss function.
        epochs (int, optional): Number of training epochs. Default is 30.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = (batch_data.to(device), batch_labels.to(device))

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model

def test_model(model, val_loader, device, criterion):
    """
    Test a PyTorch model and calculate accuracy.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for testing.
        criterion (torch.nn.Module): Loss function.

    Returns:
        dict: A dictionary containing validation metrics (accuracy, loss, recall, f1_score).
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = (
                batch_data.to(device),
                batch_labels.to(device),
            )

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct_predictions / total_samples
    recall = 100 * recall_score(all_labels, all_preds, average='weighted')
    f1 = 100 * f1_score(all_labels, all_preds, average='weighted')

    print(f'Validation Accuracy: {accuracy:.2f}%, Loss: {total_loss / len(val_loader):.4f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n')

    return {
        'accuracy': accuracy,
        'loss': total_loss / len(val_loader),
        'recall': recall,
        'f1_score': f1
    }

from sklearn.model_selection import StratifiedShuffleSplit
def cnn_validation(model, X, y, partition_ratio=0.8, partitioning="sequential",
                   batch_size=128, epochs=30, learning_rate=5e-4, random_seed=42,
                   stratified=False, stratified_shuffle=False, standardize=False):
    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------
    if not 0 < partition_ratio < 1:
        raise ValueError(
            "partition_ratio must be between 0 and 1."
        )

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.as_tensor(
        y,
        dtype=torch.long
    ).view(-1)

    if len(X_tensor) != len(y_tensor):
        raise ValueError(
            f"X and y must contain the same number of samples, "
            f"but got {len(X_tensor)} and {len(y_tensor)}."
        )

    if len(X_tensor) < 2:
        raise ValueError(
            "At least two samples are required."
        )

    # Add channel dimension:
    # (N, H, W) -> (N, 1, H, W)
    if X_tensor.ndim == 3:
        X_tensor = X_tensor.unsqueeze(1)

    # elif X_tensor.ndim != 4:
    #     raise ValueError(
    #         "X must have shape (N, H, W) or (N, C, H, W), "
    #         f"but received {tuple(X_tensor.shape)}."
    #     )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Running Device: {device}\n")

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------
    set_seed(random_seed)

    total_len = len(X_tensor)

    training_len = int(
        total_len * partition_ratio
    )

    if training_len == 0 or training_len == total_len:
        raise ValueError(
            "partition_ratio produces an empty "
            "training or validation set."
        )

    # --------------------------------------------------
    # Dataset partitioning
    # --------------------------------------------------
    if stratified:

        labels_np = y_tensor.cpu().numpy()

        # ----------------------------------------------
        # Stratified + randomized
        # ----------------------------------------------
        if stratified_shuffle:

            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=partition_ratio,
                random_state=random_seed,
            )

            training_indices, testing_indices = next(
                splitter.split(
                    np.zeros(total_len),
                    labels_np
                )
            )

            training_indices = torch.as_tensor(
                training_indices,
                dtype=torch.long
            )

            testing_indices = torch.as_tensor(
                testing_indices,
                dtype=torch.long
            )

        # ----------------------------------------------
        # Stratified + sequential
        # ----------------------------------------------
        else:

            training_indices = []
            testing_indices = []

            classes = torch.unique(
                y_tensor,
                sorted=True
            )

            for cls in classes:

                # Indices belonging to this class
                cls_indices = torch.where(
                    y_tensor == cls
                )[0]

                cls_len = len(cls_indices)

                cls_training_len = int(
                    cls_len * partition_ratio
                )

                # Make sure both train and validation
                # contain samples of this class
                if cls_training_len == 0:
                    raise ValueError(
                        f"Class {cls.item()} has too few samples "
                        f"for partition_ratio={partition_ratio}."
                    )

                if cls_training_len == cls_len:
                    raise ValueError(
                        f"Class {cls.item()} would have no "
                        f"validation samples with "
                        f"partition_ratio={partition_ratio}."
                    )

                training_indices.append(
                    cls_indices[:cls_training_len]
                )

                testing_indices.append(
                    cls_indices[cls_training_len:]
                )

            training_indices = torch.cat(
                training_indices
            )

            testing_indices = torch.cat(
                testing_indices
            )

            # Restore global/original sample order.
            #
            # Without sorting:
            # [class0 indices, class1 indices, class2 indices]
            #
            # With sorting:
            # indices remain in their original dataset order.
            training_indices = torch.sort(
                training_indices
            ).values

            testing_indices = torch.sort(
                testing_indices
            ).values

    # --------------------------------------------------
    # Non-stratified splitting
    # --------------------------------------------------
    else:

        if partitioning == "sequential":

            training_indices = torch.arange(
                0,
                training_len
            )

            testing_indices = torch.arange(
                training_len,
                total_len
            )

        elif partitioning == "randomized":

            split_generator = torch.Generator()
            split_generator.manual_seed(
                random_seed
            )

            indices = torch.randperm(
                total_len,
                generator=split_generator
            )

            training_indices = indices[
                :training_len
            ]

            testing_indices = indices[
                training_len:
            ]

        else:
            raise ValueError(
                "Invalid partitioning method. "
                "Use 'sequential' or 'randomized'."
            )

    # --------------------------------------------------
    # Extract training and validation data
    # --------------------------------------------------
    X_train = X_tensor[
        training_indices
    ]

    X_val = X_tensor[
        testing_indices
    ]

    y_train = y_tensor[
        training_indices
    ]

    y_val = y_tensor[
        testing_indices
    ]

    # --------------------------------------------------
    # Standardization
    #
    # StandardScaler is fitted ONLY on training data.
    #
    # (N, C, H, W)
    #       ↓
    # (N, C * H * W)
    #       ↓
    # StandardScaler
    #       ↓
    # reshape back to (N, C, H, W)
    # --------------------------------------------------
    if standardize:

        train_shape = X_train.shape
        val_shape = X_val.shape

        X_train_flat = X_train.reshape(
            len(X_train),
            -1
        ).cpu().numpy()

        X_val_flat = X_val.reshape(
            len(X_val),
            -1
        ).cpu().numpy()

        scaler = StandardScaler()

        X_train_flat = scaler.fit_transform(
            X_train_flat
        )

        X_val_flat = scaler.transform(
            X_val_flat
        )

        X_train = torch.as_tensor(
            X_train_flat,
            dtype=torch.float32
        ).reshape(
            train_shape
        )

        X_val = torch.as_tensor(
            X_val_flat,
            dtype=torch.float32
        ).reshape(
            val_shape
        )

    # --------------------------------------------------
    # Construct datasets
    # --------------------------------------------------
    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------
    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    # --------------------------------------------------
    # Copy model
    # --------------------------------------------------
    model = copy.deepcopy(
        model
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    model = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    metrics = test_model(
        model=model,
        val_loader=val_loader,
        device=device,
        criterion=criterion
    )

    # --------------------------------------------------
    # Information
    # --------------------------------------------------
    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Stratified: "
        f"{stratified}"
    )

    if stratified:
        print(
            f"Stratified shuffle: "
            f"{stratified_shuffle}"
        )

    print(
        f"Standardized: "
        f"{standardize}"
    )

    print(
        f"Validation Results: "
        f"{metrics}\n"
    )

    return metrics

def cnn_validation_(model, X, y, partition_ratio=0.8, partitioning="sequential",
                   batch_size=128, epochs=30, learning_rate=5e-4, random_seed=42,):
    """
    Perform cross-validation for a CNN model.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        X (array-like or torch.Tensor): 2D input data: samples x m x n or samples x channels x m x n
        y (array-like or torch.Tensor): Labels.
        partition_ratio (float, optional): Ratio of training to testing set. Default is 0.7.
        partitioning (str, optional): Type of partitioning ('sequential' or 'randomized'). Default is 'sequential'.
        batch_size (int, optional): Batch size for DataLoader. Default is 128.
        epochs (int, optional): Number of training epochs. Default is 30.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.0005.

    Returns:
        dict: A dictionary containing validation metrics (accuracy, loss, recall, f1_score).
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running Device: {device}\n')

    if partitioning == 'sequential':
        total_len = len(X_tensor)
        training_len = int(total_len * partition_ratio)
        # testing_len = total_len - training_len

        training_indices = list(range(training_len))
        testing_indices = list(range(training_len, total_len))

    elif partitioning == 'randomized':
        indices = torch.randperm(len(X_tensor))
        training_len = int(len(X_tensor) * partition_ratio)
        training_indices = indices[:training_len].tolist()
        testing_indices = indices[training_len:].tolist()

    else:
        raise ValueError("Invalid partitioning method. Use 'sequential' or 'randomized'.")

    train_dataset = TensorDataset(X_tensor[training_indices], y_tensor[training_indices])
    test_dataset = TensorDataset(X_tensor[testing_indices], y_tensor[testing_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model = train_model(model, train_loader, device, optimizer, criterion, epochs=epochs)
    metrics = test_model(model, val_loader, device, criterion)

    print(f'Validation Results: {metrics}\n')
    return metrics

def cnn_cross_validation(model, X, y, folds=5, batch_size=128, epochs=30,
                         learning_rate=5e-4, seed=42, stratified=False,
                         stratified_shuffle=False):
    """
    Perform k-fold cross-validation on a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        X (array-like or torch.Tensor): Input data: samples x m x n
        y (array-like or torch.Tensor): Labels.
        folds (int, optional): Number of folds for cross-validation. Default is 5.
        batch_size (int, optional): Batch size for DataLoader. Default is 128.
        epochs (int, optional): Number of training epochs. Default is 30.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.0005.
        seed (int, optional): Base random seed. Default is 42.
        stratified (bool, optional): If True, construct folds with
            StratifiedKFold so that class proportions are approximately
            preserved in every validation fold. If False, preserve the
            original sequential K-fold split. Default is False.
        stratified_shuffle (bool, optional): Shuffle samples before
            stratification. This only applies when ``stratified=True``.
            Default is False, which preserves the within-class sample order.

    Returns:
        dict: A dictionary containing average metrics across all folds.
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running Device: {device}\n')

    results = []

    total_len = len(X_tensor)
    if folds < 2 or folds > total_len:
        raise ValueError("folds must be between 2 and the number of samples.")

    if stratified:
        labels_np = y_tensor.cpu().numpy()
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=stratified_shuffle,
            random_state=seed if stratified_shuffle else None,
        )
        split_indices = list(splitter.split(np.zeros(total_len), labels_np))
    else:
        fold_size = total_len // folds
        indices = list(range(total_len))
    
    initial_state = copy.deepcopy(model.state_dict())
    
    for fold in range(folds):
        print(f"Fold {fold + 1}/{folds}")
        model.load_state_dict(initial_state)
        
        if stratified:
            train_idx, val_idx = split_indices[fold]
            train_idx = train_idx.tolist()
            val_idx = val_idx.tolist()
        else:
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < folds - 1 else total_len
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        
        # Random seed
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, generator=generator,
                                  num_workers=0, pin_memory=device.type=="cuda",)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=0, pin_memory=device.type == "cuda",)
        
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training Fold {fold + 1}...")
        model = train_model(model, train_loader, device, optimizer, criterion, epochs=epochs)
        metrics = test_model(model, val_loader, device, criterion)
        results.append(metrics)

    avg_metrics = {
        'accuracy': sum(r['accuracy'] for r in results) / folds,
        'f1_score': sum(r['f1_score'] for r in results) / folds,
        'loss': sum(r['loss'] for r in results) / folds,
        'recall': sum(r['recall'] for r in results) / folds,
    }

    print(f"Average Metrics Across {folds} Folds: {avg_metrics}\n")
    return avg_metrics

# %% SVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def svm_validation(X, y, valid_type="cross_validation", # hold_one_out_validation
                   folds=5, leave_part=0.4,
                   kernel="linear", C=1.0, gamma="scale", seed=42,
                   stratified=False, stratified_shuffle=False,
                   standardize=False):
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    y_np = y_np.reshape(-1)

    # Flatten input data
    X_np = X_np.reshape(len(X_np), -1)

    print('Running Device: CPU (SVM)\n')

    results = []

    total_len = len(X_np)

    if valid_type == "cross_validation":
        if folds < 2 or folds > total_len:
            raise ValueError("folds must be between 2 and the number of samples.")

        if stratified:
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=stratified_shuffle,
                random_state=seed if stratified_shuffle else None,
            )
            split_indices = list(splitter.split(np.zeros(total_len), y_np))
        else:
            fold_size = total_len // folds
            indices = list(range(total_len))

        for fold in range(folds):
            print(f"Fold {fold + 1}/{folds}")

            if stratified:
                train_idx, val_idx = split_indices[fold]
                train_idx = train_idx.tolist()
                val_idx = val_idx.tolist()
            else:
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < folds - 1 else total_len
                val_idx = indices[val_start:val_end]
                train_idx = indices[:val_start] + indices[val_end:]

            X_train = X_np[train_idx]
            X_val = X_np[val_idx]

            y_train = y_np[train_idx]
            y_val = y_np[val_idx]

            if standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed)

            print(f"Training Fold {fold + 1}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_score': f1_score(y_val, y_pred, average='macro', zero_division=0),
                'loss': None,
                'recall': recall_score(y_val, y_pred, average='macro', zero_division=0),
            }

            results.append(metrics)

    elif valid_type == "hold_one_out_validation":
        if not 0 < leave_part < 1:
            raise ValueError("leave_part must be between 0 and 1.")

        indices = np.arange(total_len)

        if stratified:
            train_idx, val_idx = train_test_split(indices,
                                                  test_size=leave_part,
                                                  random_state=None, shuffle=False)
        else:
            val_size = int(np.ceil(total_len * leave_part))
            val_idx = indices[-val_size:]
            train_idx = indices[:-val_size]

        X_train = X_np[train_idx]
        X_val = X_np[val_idx]

        y_train = y_np[train_idx]
        y_val = y_np[val_idx]

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed)

        print("Training SVM...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, average='macro', zero_division=0),
            'loss': None,
            'recall': recall_score(y_val, y_pred, average='macro', zero_division=0),
        }

        results.append(metrics)

    else:
        raise ValueError(
            "valid_type must be 'cross_validation' or 'leave_part_validation'."
        )

    avg_metrics = {
        'accuracy': sum(r['accuracy'] for r in results) / len(results),
        'f1_score': sum(r['f1_score'] for r in results) / len(results),
        'loss': None,
        'recall': sum(r['recall'] for r in results) / len(results),
    }

    if valid_type == "cross_validation":
        print(f"Average Metrics Across {folds} Folds: {avg_metrics}\n")
    else:
        print(f"Leave-Part Validation Metrics: {avg_metrics}\n")

    return avg_metrics

