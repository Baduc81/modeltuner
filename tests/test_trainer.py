import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from bartphobeit.trainer import ImprovedVQATrainer
from bartphobeit.model import ImprovedVietnameseVQAModel
from bartphobeit.config import get_improved_config

class DummyDataset(Dataset):
    def __len__(self): return 4
    def __getitem__(self, idx):
        pad_id = 1   # MBART/BARTPho pad_token_id
        eos_id = 2   # MBART/BARTPho eos_token_id (thường là 2)

        # tạo answer_ids toàn pad nhưng đảm bảo token đầu là eos
        answer_ids = torch.full((5,), pad_id, dtype=torch.long)
        answer_ids[0] = eos_id

        return {
            "pixel_values": torch.randn(3,224,224),
            "question_input_ids": torch.ones(8, dtype=torch.long),
            "question_attention_mask": torch.ones(8, dtype=torch.long),
            "answer_input_ids": answer_ids,
            "answer_attention_mask": torch.ones(5, dtype=torch.long),
            "answer_text": "mèo",
            "all_correct_answers": ["mèo", "con mèo"]
        }

@pytest.fixture
def trainer():
    cfg = get_improved_config()
    cfg["use_vqkd"] = False
    cfg["num_epochs"] = 1
    model = ImprovedVietnameseVQAModel(cfg)
    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=2)
    return ImprovedVQATrainer(model, dl, dl, "cpu", cfg)

def test_trainer_one_step(trainer):
    batch = next(iter(trainer.train_loader))
    outputs = trainer.model(
        pixel_values=batch["pixel_values"],
        question_input_ids=batch["question_input_ids"],
        question_attention_mask=batch["question_attention_mask"],
        answer_input_ids=batch["answer_input_ids"],
        answer_attention_mask=batch["answer_attention_mask"]
    )
    loss = outputs.loss
    assert loss.requires_grad
