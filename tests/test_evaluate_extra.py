from bartphobeit import evaluate

def test_metrics_all_available():
    class DummyTokenizer:
        def batch_decode(self, ids, **kw):
            return ["mèo"]

    evaluator = evaluate.ComprehensiveVQAEvaluator(DummyTokenizer())

    preds = ["mèo"]
    gts = [["mèo"]]

    # Gọi đúng hàm có thật trong class
    metrics = evaluator.calculate_comprehensive_metrics(preds, gts)

    # Phải trả về dict có key 'exact_match'
    assert isinstance(metrics, dict)
    assert "exact_match" in metrics
    assert metrics["exact_match"] == 1.0
