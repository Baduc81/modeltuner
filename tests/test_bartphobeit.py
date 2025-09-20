import torch
from bartphobeit.BARTphoBEIT import VietnameseVQADataset, VietnameseVQAModel, VQAEvaluator
from transformers import AutoTokenizer, ViTImageProcessor

def test_dataset_sample(tmp_path):
    from PIL import Image
    img_path = tmp_path / "dummy.jpg"
    Image.new("RGB", (224, 224)).save(img_path)

    questions = [{"image_name": "dummy.jpg", "question": "Con vật này là gì?", "ground_truth": "mèo"}]
    q_tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
    a_tok = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    feat = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    ds = VietnameseVQADataset(questions, str(tmp_path), q_tok, a_tok, feat)
    sample = ds[0]
    assert "pixel_values" in sample
    assert sample["answer_text"] == "mèo"

def test_model_forward_and_generate():
    cfg = {
        "vision_model": "google/vit-base-patch16-224-in21k",
        "text_model": "vinai/phobert-base",
        "decoder_model": "vinai/bartpho-syllable",
        "hidden_dim": 128,
        "freeze_encoders": True,
    }
    model = VietnameseVQAModel(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["decoder_model"])

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    batch = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "question_input_ids": torch.ones(1, 8, dtype=torch.long),
        "question_attention_mask": torch.ones(1, 8, dtype=torch.long),
        # Giả lập: [bos_token_id, some_id, eos_token_id, pad, pad]
        "answer_input_ids": torch.tensor([[tokenizer.bos_token_id, 100, eos_token_id, pad_token_id, pad_token_id]]),
        "answer_attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    outputs = model(**batch)
    assert outputs.loss is not None


def test_evaluator_normalization():
    ev = VQAEvaluator(None)
    assert ev.normalize_answer("MÈO !!!") == "mèo"
