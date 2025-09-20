import torch
from bartphobeit.model import ImprovedVietnameseVQAModel
from bartphobeit.config import get_improved_config

def test_model_with_flags():
    cfg = get_improved_config()
    cfg["use_vqkd"] = True
    cfg["use_unified_masking"] = True
    cfg["num_multiway_layers"] = 2
    model = ImprovedVietnameseVQAModel(cfg)

    pixel_values = torch.randn(1,3,224,224)
    q_ids = torch.ones(1,8, dtype=torch.long)
    q_mask = torch.ones(1,8, dtype=torch.long)
    # tránh pad_token_id=1 → dùng token id khác
    a_ids = torch.full((1,5), 2, dtype=torch.long)
    a_mask = torch.ones(1,5, dtype=torch.long)

    out = model(pixel_values, q_ids, q_mask, a_ids, a_mask)
    assert hasattr(out, "loss")
