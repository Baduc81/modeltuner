import os
from bartphobeit.model import ImprovedVietnameseVQAModel
from bartphobeit.dataset import EnhancedVietnameseVQADataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel
import json
from contextlib import nullcontext
from torch import autocast
from bartphobeit.model import normalize_vietnamese_answer
from tqdm import tqdm
import torch

# Import OpenViVQA evaluation metrics
try:
    from OpenViVQA.evaluation.accuracy import Accuracy
    from OpenViVQA.evaluation.bleu import Bleu
    from OpenViVQA.evaluation.meteor import Meteor
    from OpenViVQA.evaluation.rouge import Rouge
    from OpenViVQA.evaluation.cider import Cider
    from OpenViVQA.evaluation.precision import Precision
    from OpenViVQA.evaluation.recall import Recall
    from OpenViVQA.evaluation.f1 import F1
    OPEN_VIVQA_AVAILABLE = True
except ImportError:
    print("OpenViVQA not available. Install/clone repo: git clone https://github.com/thu-coai/OpenViVQA")
    OPEN_VIVQA_AVAILABLE = False


# ✅ FIX: WUPS metric class (thêm fallback nếu thiếu tokenizer/model)
class WUPS:
    """WUPS metric using PhoBERT embeddings or exact match"""
    def __init__(self, threshold=0.0, tokenizer=None, text_model_name=None, device='cpu'):
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.device = device
        try:
            self.text_model = AutoModel.from_pretrained(text_model_name).to(device) if text_model_name else None
            print(f"WUPS: Initialized PhoBERT for embeddings (threshold={threshold})")
        except Exception as e:
            print(f"WUPS: Failed to load PhoBERT ({e}), using exact match fallback")
            self.text_model = None

    def compute_score(self, gts, res):
        """Compute WUPS score: max cosine similarity over ground truths"""
        scores = []
        for qid in gts:
            pred = res[qid][0]
            gts_list = gts[qid]

            if not pred.strip() or not any(gt.strip() for gt in gts_list):
                scores.append(0.0)
                continue

            if self.text_model and self.tokenizer and self.threshold < 1.0:
                try:
                    with torch.no_grad():
                        with autocast("cuda", dtype=torch.float16) if self.device == 'cuda' else nullcontext():
                            inputs = self.tokenizer(
                                [pred] + gts_list,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=128
                            )
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            embeddings = self.text_model(**inputs).last_hidden_state[:, 0, :]  # CLS token
                            pred_emb = embeddings[0]
                            gt_embs = embeddings[1:]

                            cos_sim = torch.nn.functional.cosine_similarity(pred_emb.unsqueeze(0), gt_embs, dim=-1)
                            max_sim = cos_sim.max().item()
                            score = max_sim if max_sim >= self.threshold else 0.0
                except Exception as e:
                    print(f"⚠️ WUPS embedding failed, fallback to exact match: {e}")
                    score = 1.0 if any(pred == gt for gt in gts_list) else 0.0
            else:
                # ✅ FIX: fallback safe mode
                score = 1.0 if any(pred == gt for gt in gts_list) else 0.0

            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0, scores


class ComprehensiveVQAEvaluator:
    """Comprehensive VQA evaluator using OpenViVQA metrics and WUPS"""

    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config or {}

        # Initialize metrics
        self.metrics = {}
        if OPEN_VIVQA_AVAILABLE:
            self.metrics.update({
                'accuracy': Accuracy(),
                'bleu_1': Bleu(n=1),
                'bleu_2': Bleu(n=2),
                'bleu_3': Bleu(n=3),
                'bleu_4': Bleu(n=4),
                'meteor': Meteor(),
                'rouge': Rouge(),
                'cider': Cider(),
                'precision': Precision(),
                'recall': Recall(),
                'f1': F1()
            })

        # ✅ FIX: fallback AutoTokenizer nếu tokenizer=None
        safe_tokenizer = tokenizer
        if safe_tokenizer is None and self.config.get('text_model'):
            try:
                safe_tokenizer = AutoTokenizer.from_pretrained(self.config['text_model'])
            except Exception as e:
                print(f"⚠️ Could not load tokenizer for WUPS: {e}")
                safe_tokenizer = None

        self.metrics.update({
            'wups_0.0': WUPS(
                threshold=1.0,
                tokenizer=safe_tokenizer,
                text_model_name=self.config.get('text_model'),
                device=self.config.get('device', 'cpu')
            ),
            'wups_0.9': WUPS(
                threshold=0.9,
                tokenizer=safe_tokenizer,
                text_model_name=self.config.get('text_model'),
                device=self.config.get('device', 'cpu')
            )
        })

        if not OPEN_VIVQA_AVAILABLE:
            print("Using fallback custom metrics (no OpenViVQA) + WUPS")

 
    def normalize_answer(self, answer):
        """Normalize answer text for evaluation (Vietnamese-specific)"""
        return normalize_vietnamese_answer(answer)
    
    def prepare_for_evaluation(self, predictions, ground_truths):
        """Prepare predictions and ground truths in format required by OpenViVQA/WUPS"""
        gts = {}
        res = {}
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            question_id = str(i)
            pred_normalized = self.normalize_answer(pred)
            
            # Multi-GT support
            if isinstance(gt, list):
                gts[question_id] = [self.normalize_answer(g) for g in gt]
            else:
                gts[question_id] = [self.normalize_answer(gt)]
            
            res[question_id] = [pred_normalized]
        
        return res, gts
    
    def calculate_comprehensive_metrics(self, predictions, ground_truths):
        """Calculate all VQA metrics including WUPS"""
        print("Preparing data for evaluation...")
        res, gts = self.prepare_for_evaluation(predictions, ground_truths)
        
        results = {}
        
        print("Calculating metrics...")
        
        # Calculate each metric
        for metric_name, metric_evaluator in self.metrics.items():
            try:
                print(f"Computing {metric_name}...")
                
                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'wups_0.0', 'wups_0.9']:
                    score, scores = metric_evaluator.compute_score(gts, res)
                else:
                    score, scores = metric_evaluator.compute_score(gts, res)
                
                if isinstance(score, (list, tuple)):
                    results[metric_name] = float(score[0]) if len(score) > 0 else 0.0
                else:
                    results[metric_name] = float(score)
                    
                print(f"{metric_name}: {results[metric_name]:.4f}")
                
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                results[metric_name] = 0.0
        
        # Fallback custom metrics nếu OpenViVQA miss
        if not OPEN_VIVQA_AVAILABLE:
            from bartphobeit.trainer import ImprovedVQATrainer
            trainer_mock = ImprovedVQATrainer(None, None, None, 'cpu', {})
            custom_results = trainer_mock.calculate_comprehensive_metrics(predictions, ground_truths)
            results.update({k: v for k, v in custom_results.items() if k not in ['exact_match', 'total_samples', 'empty_predictions']})
        
        # Additional custom metrics
        results['exact_match'] = self.calculate_exact_match(predictions, ground_truths)
        results['total_samples'] = len(predictions)
        results['empty_predictions'] = sum(1 for pred in predictions if not pred.strip())
        
        if 'cider' in results:
            print(f"CIDEr (OpenViVQA standard): {results['cider']:.4f}")
        # THÊM: Print WUPS
        print(f"WUPS-0.0 (strict): {results.get('wups_0.0', 0.0):.4f}")
        print(f"WUPS-0.9 (loose): {results.get('wups_0.9', 0.0):.4f}")
        
        return results
    
    def calculate_exact_match(self, predictions, ground_truths):
        """Calculate exact match accuracy (multi-GT support)"""
        exact_matches = 0
        for pred, gt in zip(predictions, ground_truths):
            pred_norm = self.normalize_answer(pred)
            if isinstance(gt, list):
                exact_matches += any(pred_norm == self.normalize_answer(g) for g in gt)
            else:
                exact_matches += (pred_norm == self.normalize_answer(gt))
        return exact_matches / len(predictions)

def evaluate_model(model_path, test_questions, config, load_pretrained=True):
    """Evaluate trained model with comprehensive metrics"""
    model = ImprovedVietnameseVQAModel(config)
    
    if load_pretrained and model_path and os.path.exists(model_path):
        print(f"Loading finetuned model from {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} (normal for new components)")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)} (normal for updates)")
                
            print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Continuing with base model...")
    else:
        print("Using base model (no finetuned weights loaded)")
    
    model = model.to(config['device'])
    model.eval()
    
    # Prepare test dataset
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = AutoTokenizer.from_pretrained(config['decoder_model'])
    feature_extractor = ViTImageProcessor.from_pretrained(config['vision_model'])
    
    test_dataset = EnhancedVietnameseVQADataset(
        test_questions, config['image_dir'], question_tokenizer, answer_tokenizer, feature_extractor,
        config['max_length'], use_augmentation=False, use_multiple_answers=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # THAY ĐỔI: Pass config cho WUPS (text_model, device)
    evaluator = ComprehensiveVQAEvaluator(answer_tokenizer, config)
    
    predictions = []
    ground_truths = []
    sample_questions = []
    sample_images = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config['device'])  # THAY ĐỔI: Fix device (config thay self)
            
            try:
                with autocast("cuda", dtype=torch.float16) if config['device'] == 'cuda' else nullcontext():
                    generated_ids = model(
                        pixel_values=batch['pixel_values'],
                        question_input_ids=batch['question_input_ids'],
                        question_attention_mask=batch['question_attention_mask']
                    )
                
                batch_predictions = answer_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                if 'all_correct_answers' in batch:
                    ground_truths.extend(batch['all_correct_answers'])
                else:
                    ground_truths.extend([ans] for ans in batch['answer_text'])
                
                sample_questions.extend(batch['question_text'])
                
                if batch_idx < 3:
                    sample_images.extend(batch.get('image_name', [f'batch_{batch_idx}'] * len(batch_predictions)))
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                batch_size = len(batch['answer_text'])
                predictions.extend([""] * batch_size)
                ground_truths.extend([ans] for ans in batch['answer_text'])
                sample_questions.extend(batch['question_text'])
    
    print(f"\nCompleted evaluation on {len(predictions)} samples")
    
    # Calculate comprehensive metrics
    metrics = evaluator.calculate_comprehensive_metrics(predictions, ground_truths)
    
    # Print results
    print_evaluation_results(metrics, load_pretrained and model_path and os.path.exists(model_path))
    
    # Print sample results
    print_sample_results(sample_questions, predictions, ground_truths, sample_images)
    
    return metrics, predictions, ground_truths

def print_evaluation_results(metrics, is_finetuned):
    """Print comprehensive evaluation results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE VQA EVALUATION RESULTS")
    print("="*80)
    
    print(f"Model type: {'Finetuned' if is_finetuned else 'Base (no finetuning)'}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Empty predictions: {metrics['empty_predictions']} ({metrics['empty_predictions']/metrics['total_samples']*100:.2f}%)")
    
    print("\n--- CORE METRICS ---")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    # THÊM: WUPS
    print(f"WUPS-0.0 (strict): {metrics['wups_0.0']:.4f}")
    print(f"WUPS-0.9 (loose): {metrics['wups_0.9']:.4f}")
    
    print("\n--- TEXT GENERATION METRICS ---")
    print(f"BLEU-1: {metrics['bleu_1']:.4f}")
    print(f"BLEU-2: {metrics['bleu_2']:.4f}")
    print(f"BLEU-3: {metrics['bleu_3']:.4f}")
    print(f"BLEU-4: {metrics['bleu_4']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    print(f"ROUGE: {metrics['rouge']:.4f}")
    print(f"CIDEr: {metrics['cider']:.4f}")
    
    print("="*80)

def print_sample_results(questions, predictions, ground_truths, images=None, num_samples=10):
    """Print sample results with detailed analysis"""
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    for i in range(min(num_samples, len(questions))):
        print(f"\nSample {i+1}:")
        if images and i < len(images):
            print(f"Image: {images[i]}")
        print(f"Question: {questions[i]}")
        print(f"Predicted: '{predictions[i]}'")
        
        # THÊM: Print multi-GT nếu list
        if isinstance(ground_truths[i], list):
            print("Ground Truths: " + ", ".join([f'"{gt}"' for gt in ground_truths[i]]))
        else:
            print(f"Ground Truth: '{ground_truths[i]}'")
        
        # Detailed comparison
        pred_norm = predictions[i].strip().lower()
        if isinstance(ground_truths[i], list):
            is_exact_match = any(pred_norm == gt.strip().lower() for gt in ground_truths[i])
            is_partial_match = any(pred_norm in gt.strip().lower() or gt.strip().lower() in pred_norm for gt in ground_truths[i])
        else:
            gt_norm = ground_truths[i].strip().lower()
            is_exact_match = pred_norm == gt_norm
            is_partial_match = pred_norm in gt_norm or gt_norm in pred_norm
        is_empty = not pred_norm.strip()
        
        if is_exact_match:
            result = "✓ EXACT MATCH"
        elif is_partial_match:
            result = "~ PARTIAL MATCH"
        elif is_empty:
            result = "✗ EMPTY PREDICTION"
        else:
            result = "✗ NO MATCH"
            
        print(f"Result: {result}")
        print("-" * 60)
    
    print(f"\nShowing {min(num_samples, len(questions))} out of {len(questions)} total samples")
    print("="*80)

def analyze_answer_distribution(predictions, ground_truths):
    """Analyze the distribution of answers"""
    from collections import Counter
    
    print("\n" + "="*80)
    print("ANSWER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Length analysis
    pred_lengths = [len(pred.split()) for pred in predictions if pred.strip()]
    truth_lengths = []
    for gt in ground_truths:
        if isinstance(gt, list):
            truth_lengths.extend([len(g.split()) for g in gt if g.strip()])
        else:
            if gt.strip():
                truth_lengths.append(len(gt.split()))
    
    print(f"Average predicted answer length: {sum(pred_lengths)/len(pred_lengths):.2f} words" if pred_lengths else "No valid predictions")
    print(f"Average ground truth answer length: {sum(truth_lengths)/len(truth_lengths):.2f} words" if truth_lengths else "No valid ground truths")
    
    if pred_lengths:
        print(f"Max predicted answer length: {max(pred_lengths)} words")
    if truth_lengths:
        print(f"Max ground truth answer length: {max(truth_lengths)} words")
    
    # Most common answers
    pred_counter = Counter([pred.strip().lower() for pred in predictions if pred.strip()])
    truth_counter = Counter()
    for gt in ground_truths:
        if isinstance(gt, list):
            truth_counter.update([g.strip().lower() for g in gt if g.strip()])
        else:
            if gt.strip():
                truth_counter.update([gt.strip().lower()])
    
    print("\nTop 10 most common predictions:")
    for answer, count in pred_counter.most_common(10):
        print(f"  '{answer}': {count} ({count/len(predictions)*100:.1f}%)")
    
    print("\nTop 10 most common ground truths:")
    for answer, count in truth_counter.most_common(10):
        print(f"  '{answer}': {count} ({count/sum(truth_counter.values())*100:.1f}%)")
    
    print("="*80)