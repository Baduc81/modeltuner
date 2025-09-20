import torch
from torch.utils.data import DataLoader, Dataset
from bartphobeit.trainer import ImprovedVQATrainer
from bartphobeit.model import ImprovedVietnameseVQAModel
from bartphobeit.config import get_improved_config

class SmallDataset(Dataset):
    def __len__(self): return 2
    def __getitem__(self, idx):
        ans_ids = torch.ones(5, dtype=torch.long)
        ans_ids[0] = 2  # tránh toàn pad
        return {
            "pixel_values": torch.randn(3,224,224),
            "question_input_ids": torch.ones(8, dtype=torch.long),
            "question_attention_mask": torch.ones(8, dtype=torch.long),
            "answer_input_ids": ans_ids,
            "answer_attention_mask": torch.ones(5, dtype=torch.long),
            "answer_text": "mèo",
            "all_correct_answers": ["mèo"]
        }

def test_trainer_full(monkeypatch, tmp_path):
    cfg = get_improved_config()
    cfg["num_epochs"] = 1
    cfg["use_vqkd"] = False
    model = ImprovedVietnameseVQAModel(cfg)
    ds = SmallDataset()
    dl = DataLoader(ds, batch_size=1)

    trainer = ImprovedVQATrainer(model, dl, dl, "cpu", cfg)
    trainer.checkpoint_dir = tmp_path

    loss = trainer.train_epoch(epoch=0)
    assert isinstance(loss, float)

    # Không có evaluate -> chỉ test train() trả về số / loss tốt nhất
    best = trainer.train(1)
    assert isinstance(best, (int, float))
