import torch
from bartphobeit.BARTphoBEIT import VQATrainer

def test_vqa_trainer(monkeypatch):
    # Fake model với tham số giả để optimizer không lỗi
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **kwargs):
            return type("out", (), {"loss": torch.tensor(0.5, requires_grad=True)})

        class decoder_tokenizer:
            @staticmethod
            def batch_decode(ids, **kwargs):
                return ["mèo"]

    # Patch compute_metrics để tránh recursion
    monkeypatch.setattr("bartphobeit.model.compute_metrics", lambda *a, **kw: {"acc": 1.0})

    ds = [{
        "pixel_values": torch.randn(3, 224, 224),
        "question_input_ids": torch.ones(8, dtype=torch.long),
        "question_attention_mask": torch.ones(8, dtype=torch.long),
        "answer_input_ids": torch.ones(5, dtype=torch.long),
        "answer_attention_mask": torch.ones(5, dtype=torch.long),
        "answer_text": "mèo",
        "all_correct_answers": [["mèo"]],
    }]
    dl = [ds[0]]

    trainer = VQATrainer(DummyModel(), dl, dl, "cpu", {"learning_rate": 1e-3})

    loss = trainer.train_epoch()
    assert isinstance(loss, float)

    metrics, preds, gts = trainer.evaluate()
    assert isinstance(metrics, dict)
    assert isinstance(preds, list)
    assert isinstance(gts, list)
