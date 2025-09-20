import pytest
import torch
from bartphobeit.dataset import EnhancedVietnameseVQADataset
from transformers import AutoTokenizer, ViTImageProcessor

@pytest.fixture
def sample_question():
    return {
        "image_name": "dummy.jpg",
        "question": "Cái này là gì?",
        "answers": ["con mèo", "mèo"],
        "ground_truth": "con mèo",
        "all_correct_answers": ["con mèo", "mèo"]
    }

@pytest.fixture
def dataset(tmp_path, sample_question):
    # Tạo 1 ảnh giả
    dummy_img = tmp_path / "dummy.jpg"
    from PIL import Image
    Image.new("RGB", (224,224), color="white").save(dummy_img)

    q_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    a_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    return EnhancedVietnameseVQADataset(
        [sample_question], str(tmp_path), q_tokenizer, a_tokenizer, extractor,
        use_augmentation=True, use_multiple_answers=True
    )

def test_dataset_item(dataset):
    item = dataset[0]
    assert "pixel_values" in item
    assert item["pixel_values"].shape[0] == 3  # RGB
    assert "question_input_ids" in item
    assert "answer_input_ids" in item
    assert isinstance(item["all_correct_answers"], list)
