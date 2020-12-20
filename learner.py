from collections import Counter
import time
from typing import Callable, Tuple, Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer, Adam
from tqdm import tqdm

from data import BobSueDataset, get_dataloader


class LanguageModelLearner:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model: nn.Module, train_set: BobSueDataset, valid_set: BobSueDataset, test_set: BobSueDataset,
                 batch_size: int, optim_cls: Callable[..., Optimizer] = Adam, lr: float = 1e-3):
        """
        Learner for training language models.

        Args:
            model: Model.
            train_set: Training dataset.
            valid_set: Validation dataset.
            test_set: Testing dataset.
            batch_size: Batch size.
            optim_cls: Optimizer class. Default: Adam
            lr: Learning rate. Default: 1e-3
        """
        self.model = model.to(self.device)
        self.train_loader = get_dataloader(train_set, batch_size=batch_size)
        self.valid_loader = get_dataloader(valid_set, batch_size=batch_size)
        self.test_loader = get_dataloader(test_set, batch_size=batch_size)
        self.optimizer = optim_cls(self.model.parameters(), lr=lr)

    def _get_metrics(self, batch: Dict[str, torch.LongTensor], return_mismatches: bool = False) \
            -> Tuple[torch.Tensor, float, int, Optional[Counter]]:
        """
        Passes a batch of data and returns metrics, such as loss,
        accuracy, etc.

        Args:
            batch: One batch data from iterating DataLoader.
            return_mismatches: Return mismatch counter if set to True.
                Default: False

        Returns:
            Loss, accuracy, batch count, mismatch counter (optional)
        """
        for k in batch:
            if not k.endswith('_lengths'):
                batch[k] = batch[k].to(self.device)

        loss, predictions, ground_truth = self.model(**batch)
        mask = predictions == ground_truth  # batch_count
        accuracy = mask.float().mean().item()

        counter = None
        if return_mismatches:
            matches = torch.vstack([predictions, ground_truth])
            mismatches = matches[:, ~mask]
            counter = Counter(map(tuple, mismatches.t().tolist()))

        return loss, accuracy, len(ground_truth), counter

    @torch.no_grad()
    def evaluate(self, valid: bool = False, return_mismatches=True) -> Tuple[float, float, Optional[Counter]]:
        """
        Evaluates metrics with validation or testing dataset.

        Args:
            valid: Use valid dataset (True) or test dataset (False).
                Default: False
            return_mismatches: Return mismatch counter if set to True.
                Default: True

        Returns:
            Loss, accuracy, mismatch counter (optional)
        """
        self.model.eval()
        data_loader = self.valid_loader if valid else self.test_loader

        sum_loss = sum_acc = total_count = 0
        counter = Counter() if return_mismatches else None
        for batch in data_loader:
            loss, acc, count, mismatches = self._get_metrics(batch, return_mismatches=return_mismatches)

            total_count += count
            sum_loss += loss.item() * count
            sum_acc += acc * count
            if return_mismatches:
                counter += mismatches

        return sum_loss / total_count, sum_acc / total_count, counter

    def _train_1_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Trains model by 1 epoch.

        Args:
            epoch: Current epoch.

        Returns:
            Loss, accuracy
        """
        self.model.train()
        loader = tqdm(self.train_loader, desc=f'Epoch {epoch + 1:02}', total=len(self.train_loader))

        sum_loss = sum_acc = total_count = 0
        for batch in loader:
            loss, acc, count, _ = self._get_metrics(batch)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_count += count
            batch_loss = loss.item()
            sum_loss += batch_loss * count
            sum_acc += acc * count
            loader.set_postfix({'Loss': batch_loss, 'Acc': acc})

        return sum_loss / total_count, sum_acc / total_count

    def train(self, epochs: int, filename: str, return_mismatches: bool = False) -> Optional[Counter]:
        """
        Trains model by multiple epochs and saves the parameters of
        the model with the lowest validation loss.

        Args:
            epochs: Number of epochs to train.
            filename: Filename to save model parameters.
            return_mismatches: Return mismatch counter if set to True.
                Default: False

        Returns:
            Mismatch counter (optional)
        """
        min_valid_loss = float('inf')
        mismatches = None

        for epoch in range(epochs):
            train_loss, train_acc = self._train_1_epoch(epoch)
            valid_loss, valid_acc, counter = self.evaluate(valid=True, return_mismatches=return_mismatches)

            print(f'\tTrain Loss: {train_loss:.3f}\tTrain Acc: {train_acc * 100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f}\tValid Acc: {valid_acc * 100:.2f}%')

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                mismatches = counter
                torch.save(self.model.state_dict(), filename)
                print(f'\tModel parameters saved to {filename}')

            time.sleep(0.5)  # avoid nested tqdm chaos

        return mismatches

    def load_model_params(self, filename: str):
        """
        Loads parameters from a file. Do nothing if parameters not
        matching exactly.

        Args:
            filename: Filename with saved model parameters.
        """
        curr_state = self.model.state_dict()
        missing_keys, unexpected_keys = self.model.load_state_dict(torch.load(filename))
        if missing_keys or unexpected_keys:
            self.model.load_state_dict(curr_state)
            raise KeyError('Parameters not matching with model, aborted.')

    def print_test_results(self):
        """
        Prints testing loss and accuracy.
        """
        test_loss, test_acc, _ = self.evaluate(return_mismatches=False)
        print(f'\t Test Loss: {test_loss:.3f}\t Test Acc: {test_acc * 100:.2f}%')
