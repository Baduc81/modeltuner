import types
import bartphobeit.main as main

def test_main_runs(monkeypatch):
    # Patch data + trainer để tránh huấn luyện nặng
    monkeypatch.setattr(main, "pd", types.SimpleNamespace(read_csv=lambda _: 
        __import__("pandas").DataFrame([
            {"image_name":"fake.png", "question":"cái gì?", "answers":["mèo"]},
            {"image_name":"fake2.png", "question":"con gì?", "answers":["chó"]},
        ])
    ))
    monkeypatch.setattr(main, "prepare_data_from_dataframe", lambda df: [
        {"image_name":"f1.png", "question":"?", "answers":["mèo"],
         "ground_truth":"mèo", "all_correct_answers":["mèo"]},
        {"image_name":"f2.png", "question":"?", "answers":["chó"],
         "ground_truth":"chó", "all_correct_answers":["chó"]},
    ])
    monkeypatch.setattr(main, "ImprovedVQATrainer", lambda *a, **kw:
        types.SimpleNamespace(train=lambda n: 0.5))

    # Run main (đã mock nặng)
    main.main()
