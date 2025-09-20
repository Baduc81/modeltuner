import pytest
from bartphobeit.evaluate import ComprehensiveVQAEvaluator
from transformers import AutoTokenizer
from bartphobeit.config import get_improved_config

@pytest.fixture
def evaluator():
    cfg = get_improved_config()
    tok = AutoTokenizer.from_pretrained(cfg["decoder_model"])
    return ComprehensiveVQAEvaluator(tok, cfg)

def test_metrics(evaluator):
    preds = ["con mèo", "con chó"]
    gts = [["mèo", "con mèo"], ["chó"]]
    results = evaluator.calculate_comprehensive_metrics(preds, gts)
    assert "accuracy" in results
    assert "wups_0.0" in results
    assert results["total_samples"] == 2
