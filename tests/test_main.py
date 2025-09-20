import runpy

def test_import_main():
    mod = runpy.run_path("bartphobeit/main.py")
    assert "main" in mod
    assert callable(mod["main"])

def test_analyze_data_balance():
    from bartphobeit.main import analyze_data_balance
    questions = [
        {"question": "Ảnh này là gì?", "ground_truth": "mèo", "all_correct_answers": ["mèo", "con mèo"]},
        {"question": "Ảnh này là gì?", "ground_truth": "chó", "all_correct_answers": ["chó", "con chó"]}
    ]
    counts = analyze_data_balance(questions)
    assert "mèo" in counts
    assert "chó" in counts
