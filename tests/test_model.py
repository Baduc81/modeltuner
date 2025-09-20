import pytest
import torch
from bartphobeit.model import ImprovedVietnameseVQAModel
from bartphobeit.config import get_improved_config

@pytest.fixture
def model():
    cfg = get_improved_config()
    cfg["use_vqkd"] = False  # disable heavy BEiT in unit test
    return ImprovedVietnameseVQAModel(cfg)

def test_forward_train(model):
    batch_size, seq_len = 2, 16
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    pad_id = model.text_decoder.config.pad_token_id
    eos_id = model.text_decoder.config.eos_token_id or 2
    answer_ids = torch.full((batch_size, 8), pad_id, dtype=torch.long)
    answer_ids[:,0] = eos_id   # ít nhất một token khác pad

    outputs = model(
        pixel_values=pixel_values,
        question_input_ids=input_ids,
        question_attention_mask=attn_mask,
        answer_input_ids=answer_ids,
        answer_attention_mask=torch.ones_like(answer_ids)
    )
    assert hasattr(outputs, "loss")

def test_forward_inference(model):
    batch_size, seq_len = 1, 16
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    generated = model(
        pixel_values=pixel_values,
        question_input_ids=input_ids,
        question_attention_mask=attn_mask
    )
    assert isinstance(generated, torch.Tensor)
    assert generated.ndim == 2
