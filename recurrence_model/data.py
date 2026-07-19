from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

ignore_label_id = 0
pad_id = 0


class TokenLabelDataset(Dataset):
    """Base dataset shape: input_ids [T], labels [T]. Label 0 is ignored."""

    vocab_size: int
    max_seq_len: int

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class SyntheticStateTrackingDataset(TokenLabelDataset):
    """Small synthetic state-tracking task for architecture smoke tests."""

    none_id = 1
    set_offset = 2

    def __init__(self, n_samples: int, seq_len: int, num_states: int, seed: int):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.num_states = num_states
        self.vocab_size = 2 + 2 * num_states
        self.max_seq_len = seq_len
        rng = random.Random(seed)
        self.examples = [self._make_example(rng) for _ in range(n_samples)]

    @property
    def add_offset(self) -> int:
        return self.set_offset + self.num_states

    def _make_example(self, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
        state = 0
        toks: List[int] = []
        for _ in range(self.seq_len):
            op_family = rng.choices(["none", "set", "add"], weights=[0.25, 0.35, 0.40])[0]
            if op_family == "none":
                tok = self.none_id
            elif op_family == "set":
                value = rng.randrange(self.num_states)
                tok = self.set_offset + value
                state = value
            else:
                value = rng.randrange(self.num_states)
                tok = self.add_offset + value
                state = (state + value) % self.num_states
            toks.append(tok)
        labels = [ignore_label_id] * self.seq_len
        labels[-1] = state + 1
        return torch.tensor(toks, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        return {"input_ids": input_ids, "labels": labels}


class SudokuExtremeDataset(TokenLabelDataset):
    """Sudoku-Extreme as 81-position token classification.

    Encoding follows TinyRecursiveModels style:
    pad=0, blank=1, digit d -> d+1.
    Labels are solution digits d+1 at all 81 positions.
    """

    def __init__(self, root: str, split: str, max_samples: Optional[int] = None):
        from datasets import load_from_disk

        ds = load_from_disk(root)[split]
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.ds = ds
        self.vocab_size = 11
        self.max_seq_len = 81

    @staticmethod
    def encode_question(text: str) -> List[int]:
        text = text.strip()
        return [1 if ch == "." else int(ch) + 1 for ch in text]

    @staticmethod
    def encode_answer(text: str) -> List[int]:
        text = text.strip()
        return [int(ch) + 1 for ch in text]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ds[idx]
        input_ids = torch.tensor(self.encode_question(row["question"]), dtype=torch.long)
        labels = torch.tensor(self.encode_answer(row["answer"]), dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class PermutationDataset(TokenLabelDataset):
    """Permutation composition as aligned chain-of-thought state tracking.

    Input: story permutation tokens.
    Labels: cumulative permutation state after each input token.
    """

    def __init__(self, root: str, subset: str, split: str, max_samples: Optional[int] = None, seed: int = 0, files_override: Optional[List[Path]] = None):
        self.data_dir = Path(root) / subset / split
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Permutation split not found: {self.data_dir}")
        files = sorted(files_override) if files_override is not None else sorted(self.data_dir.glob("*.json"))
        if max_samples is not None and max_samples < len(files):
            rng = random.Random(seed)
            files = rng.sample(files, max_samples)
            files.sort()
        self.files = files
        self.n = self._infer_n(subset, files)
        self.perm_tokens = self._all_perm_strings(self.n)
        self.token_to_id = {tok: i + 1 for i, tok in enumerate(self.perm_tokens)}
        self.vocab_size = 1 + len(self.perm_tokens)
        self.max_seq_len = self._infer_max_seq_len(files)

    @staticmethod
    def _infer_n(subset: str, files: List[Path]) -> int:
        m = re.search(r"S(\d+)_", subset)
        if m:
            return int(m.group(1))
        sample = json.loads(files[0].read_text())
        return len(sample["state_seq"][-1])

    @staticmethod
    def _all_perm_strings(n: int) -> List[str]:
        import itertools

        return ["".join(map(str, p)) for p in itertools.permutations(range(1, n + 1))]

    @staticmethod
    def _infer_max_seq_len(files: List[Path]) -> int:
        sample = json.loads(files[0].read_text())
        story = sample["story"]
        return len(story.split()) if isinstance(story, str) else len(story)

    @staticmethod
    def _state_to_token(state: List[int]) -> str:
        return "".join(map(str, state))

    def _encode_story(self, story) -> List[int]:
        if isinstance(story, str):
            return [self.token_to_id[tok] for tok in story.split()]
        return [int(tok) for tok in story]

    def _encode_state_seq(self, state_seq, story_len: int) -> List[int]:
        if len(state_seq) == story_len + 1:
            state_seq = state_seq[1:]
        if len(state_seq) != story_len:
            raise ValueError(f"Expected one state label per story token, got {len(state_seq)} states for {story_len} tokens")
        if state_seq and isinstance(state_seq[0], list):
            return [self.token_to_id[self._state_to_token(state)] for state in state_seq]
        return [int(state) for state in state_seq]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obj = json.loads(self.files[idx].read_text())
        input_ids = self._encode_story(obj["story"])
        labels = self._encode_state_seq(obj["state_seq"], len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }



class BabiDataset(TokenLabelDataset):
    """bAbI QA as decoder-only answer prediction.

    The model reads story + question + <A>. Labels are shifted so the <A>
    position predicts the first answer token, and answer-prefix positions
    predict the next answer token or <EOS>.
    """

    q_token = "<Q>"
    a_token = "<A>"
    eos_token = "<EOS>"

    def __init__(self, examples: List[Tuple[List[str], List[str]]], token_to_id: Dict[str, int], max_seq_len: Optional[int] = None):
        self.examples = examples
        self.token_to_id = token_to_id
        self.vocab_size = 1 + len(token_to_id)
        self.max_seq_len = max_seq_len if max_seq_len is not None else self._infer_max_seq_len(examples)

    @classmethod
    def from_examples(cls, examples: List[Tuple[List[str], List[str]]], token_to_id: Dict[str, int], max_seq_len: Optional[int] = None) -> "BabiDataset":
        return cls(examples, token_to_id, max_seq_len=max_seq_len)

    @classmethod
    def parse_file(cls, path: Path) -> List[Tuple[List[str], List[str]]]:
        if not path.exists():
            raise FileNotFoundError(f"bAbI file not found: {path}")
        examples: List[Tuple[List[str], List[str]]] = []
        story: List[str] = []
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            idx_text = line.split(" ", 1)
            if len(idx_text) != 2:
                continue
            line_id = int(idx_text[0])
            text = idx_text[1]
            if line_id == 1:
                story = []
            if "\t" in text:
                question, answer, _support = text.split("\t")
                context = list(story) + [cls.q_token] + cls.tokenize(question) + [cls.a_token]
                answer_tokens = cls.tokenize(answer)
                examples.append((context, answer_tokens))
            else:
                story.extend(cls.tokenize(text))
        return examples

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text.lower())

    @classmethod
    def build_vocab(cls, example_groups) -> Dict[str, int]:
        if example_groups and isinstance(example_groups[0], tuple):
            groups = [example_groups]
        else:
            groups = example_groups
        vocab = {cls.q_token, cls.a_token, cls.eos_token}
        for examples in groups:
            for context, answer in examples:
                vocab.update(context)
                vocab.update(answer)
        return {tok: idx + 1 for idx, tok in enumerate(sorted(vocab))}

    @classmethod
    def _infer_max_seq_len(cls, examples: List[Tuple[List[str], List[str]]]) -> int:
        if not examples:
            return 0
        return max(len(context) + len(answer) for context, answer in examples)

    @staticmethod
    def resolve_task_file(root: str, version: str, task: str, split: str) -> Path:
        version_dir = Path(root) / version
        if not version_dir.exists():
            raise FileNotFoundError(f"bAbI version directory not found: {version_dir}")
        matches = sorted(version_dir.glob(f"{task}_*_{split}.txt"))
        if len(matches) != 1:
            available = sorted(p.name.split("_", 1)[0] for p in version_dir.glob(f"*_{split}.txt"))
            raise FileNotFoundError(f"Expected exactly one file for task={task} split={split} under {version_dir}, found {len(matches)}. Available task ids: {available}")
        return matches[0]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        context, answer = self.examples[idx]
        target = answer + [self.eos_token]
        input_tokens = context + answer
        labels = [ignore_label_id] * len(input_tokens)
        start = len(context) - 1
        for offset, tok in enumerate(target):
            labels[start + offset] = self.token_to_id[tok]
        input_ids = [self.token_to_id[tok] for tok in input_tokens]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def split_babi_examples(examples: List[Tuple[List[str], List[str]]], max_train_samples: Optional[int], max_val_samples: Optional[int], seed: int) -> Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]]]:
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_examples = len(shuffled)
    if max_train_samples is None:
        desired_val = max_val_samples if max_val_samples is not None else min(1000, n_examples // 10)
        if desired_val >= n_examples:
            desired_val = min(1000, max(0, n_examples // 10))
        val_count = min(desired_val, max(0, n_examples - 1))
        train_count = n_examples - val_count
    else:
        train_count = min(max_train_samples, n_examples)
        remaining = max(0, n_examples - train_count)
        val_count = remaining if max_val_samples is None else min(max_val_samples, remaining)
    train_examples = shuffled[:train_count]
    val_examples = shuffled[train_count : train_count + val_count]
    return train_examples, val_examples


def collate_token_labels(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), ignore_label_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_datasets(args):
    if args.dataset == "synthetic":
        train_ds = SyntheticStateTrackingDataset(args.train_samples, args.seq_len, args.num_states, args.seed)
        val_ds = SyntheticStateTrackingDataset(args.val_samples, args.seq_len, args.num_states, args.seed + 1)
        test_ds = SyntheticStateTrackingDataset(args.val_samples, args.seq_len, args.num_states, args.seed + 1)
    elif args.dataset == "sudoku":
        from datasets import load_from_disk

        train_split = load_from_disk(args.sudoku_root)["train"]
        desired_val = min(args.max_val_samples, len(train_split))
        train_count = max(0, len(train_split) - desired_val) if args.max_train_samples is None else min(args.max_train_samples, max(0, len(train_split) - desired_val))
        val_count = min(desired_val, max(0, len(train_split) - train_count))
        train_ds = SudokuExtremeDataset(args.sudoku_root, "train", train_count)
        val_ds = SudokuExtremeDataset(args.sudoku_root, "train", train_count + val_count)
        val_ds.ds = val_ds.ds.select(range(train_count, train_count + val_count))
        test_ds = SudokuExtremeDataset(args.sudoku_root, "test", args.max_test_samples)
    elif args.dataset == "permutation":
        train_dir = Path(args.permutation_root) / args.permutation_subset / "train"
        train_files = sorted(train_dir.glob("*.json"))
        if not train_files:
            raise FileNotFoundError(f"Permutation train split not found or empty: {train_dir}")
        rng = random.Random(args.seed)
        rng.shuffle(train_files)
        desired_val = min(args.max_val_samples, len(train_files))
        train_count = max(0, len(train_files) - desired_val) if args.max_train_samples is None else min(args.max_train_samples, max(0, len(train_files) - desired_val))
        val_count = min(desired_val, max(0, len(train_files) - train_count))
        train_selected = sorted(train_files[:train_count])
        val_selected = sorted(train_files[train_count : train_count + val_count])
        train_ds = PermutationDataset(args.permutation_root, args.permutation_subset, "train", seed=args.seed, files_override=train_selected)
        val_ds = PermutationDataset(args.permutation_root, args.permutation_subset, "train", seed=args.seed + 1, files_override=val_selected)
        test_ds = PermutationDataset(args.permutation_root, args.permutation_subset, "test", args.max_test_samples, args.seed + 2)
    elif args.dataset == "babi":
        train_path = BabiDataset.resolve_task_file(args.babi_root, args.babi_version, args.babi_task, "train")
        test_path = BabiDataset.resolve_task_file(args.babi_root, args.babi_version, args.babi_task, "test")
        train_all = BabiDataset.parse_file(train_path)
        test_examples = BabiDataset.parse_file(test_path)
        train_examples, val_examples = split_babi_examples(train_all, args.max_train_samples, args.max_val_samples, args.seed)
        if args.max_test_samples is not None:
            test_examples = test_examples[: args.max_test_samples]
        token_to_id = BabiDataset.build_vocab([train_examples, val_examples, test_examples])
        max_seq_len = max(BabiDataset._infer_max_seq_len(examples) for examples in [train_examples, val_examples, test_examples] if examples)
        train_ds = BabiDataset.from_examples(train_examples, token_to_id, max_seq_len=max_seq_len)
        val_ds = BabiDataset.from_examples(val_examples, token_to_id, max_seq_len=max_seq_len)
        test_ds = BabiDataset.from_examples(test_examples, token_to_id, max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_ds, val_ds, test_ds
