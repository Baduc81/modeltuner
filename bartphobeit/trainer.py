import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import json
import glob
from collections import defaultdict
import wandb
from datetime import datetime
from bartphobeit.model import ImprovedVietnameseVQAModel, normalize_vietnamese_answer
from contextlib import nullcontext  # Thêm import cho non-FP16 context

# Install required packages for evaluation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer  # type: ignore
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_ROUGE_AVAILABLE = True
except ImportError:
    print("NLTK/Rouge not available. Install with: pip install nltk rouge-score")
    BLEU_ROUGE_AVAILABLE = False

class ImprovedVQATrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.current_stage = 1
        self.global_step = 0

        self.accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        if self.model is not None:
            self.setup_optimizers_and_schedulers()

        # For evaluation
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_fuzzy_accuracy = 0
        
        # Setup logging
        self.setup_logging()
        
        # Checkpoint management
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Evaluation metrics
        if BLEU_ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            self.smoothing_function = SmoothingFunction().method4
        
        # Thêm FP16 scaler cho mixed precision
        from torch.amp import GradScaler, autocast
        self.scaler = GradScaler("cuda") if self.config.get('use_fp16', True) else None
        print(f"FP16 enabled: {self.scaler is not None}")

        # Enable gradient checkpointing cho large models (ViT/PhoBERT)
        if self.model is not None:
            if hasattr(self.model.vision_model, 'gradient_checkpointing_enable'):
                self.model.vision_model.gradient_checkpointing_enable()
            if hasattr(self.model.text_model, 'gradient_checkpointing_enable'):
                self.model.text_model.gradient_checkpointing_enable()

        # Gradient accumulation steps từ config
        self.accumulation_steps = self.config.get('gradient_accumulation_steps', 2)
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
    
    def setup_logging(self):
        """Setup wandb logging"""
        if self.config.get('use_wandb', False):
            try:
                wandb.init(
                    project=self.config.get('project_name', 'Vietnamese-VQA'),
                    config=self.config,
                    name=f"VQA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.use_wandb = True
                print("Wandb logging initialized")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def setup_optimizers_and_schedulers(self):
        """Setup optimizers with improved scheduler"""
        
        # Group parameters by component
        decoder_params = []
        encoder_params = []
        vision_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'text_decoder' in name:
                decoder_params.append(param)
            elif 'text_model' in name:
                encoder_params.append(param)
            elif 'vision_model' in name:
                vision_params.append(param)
            else:  # fusion layer and projections
                fusion_params.append(param)
        
        # Setup parameter groups with different learning rates
        param_groups = []
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params, 
                'lr': self.config['decoder_lr'],
                'name': 'decoder'
            })
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params, 
                'lr': self.config['encoder_lr'],
                'name': 'encoder'
            })
        
        if vision_params:
            param_groups.append({
                'params': vision_params, 
                'lr': self.config['vision_lr'],
                'name': 'vision'
            })
        
        if fusion_params:
            param_groups.append({
                'params': fusion_params, 
                'lr': self.config['decoder_lr'],  # Same as decoder
                'name': 'fusion'
            })
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        
        # Setup improved scheduler with warmup + linear decay
        accum = max(1, self.accumulation_steps)
        steps_per_epoch = max(1, len(self.train_loader) // accum)
        total_steps = steps_per_epoch * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Enhanced optimizer setup:")
        print(f"  Parameter groups: {len(param_groups)}")
        print(f"  Total training steps (optimizer updates): {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")


    def evaluate_vqa(self):
        """Enhanced VQA evaluation with comprehensive metrics"""
        self.model.eval()
        predictions = []
        ground_truths = []  # Sẽ là list of lists nếu multi-GT
        questions = []
        progress_bar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Generate predictions
                generated_ids = self.model(
                    pixel_values=batch['pixel_values'],
                    question_input_ids=batch['question_input_ids'],
                    question_attention_mask=batch['question_attention_mask']
                )
                
                # Decode predictions
                pred_texts = self.model.decoder_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                predictions.extend(pred_texts)
                
                # Collect ground_truths as list of lists cho multi-GT
                if 'all_correct_answers' in batch:
                    ground_truths.extend(batch['all_correct_answers'])
                else:
                    ground_truths.extend([[ans] for ans in batch['answer_text']])  # Wrap single as list
                
                questions.extend(batch.get('question_text', [''] * len(pred_texts)))  # Giữ nguyên câu hỏi nếu có
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(predictions, ground_truths)
        
        return metrics, predictions, ground_truths, questions
    
    def calculate_comprehensive_metrics(self, predictions, ground_truths):
        """Calculate comprehensive VQA metrics including BLEU, ROUGE, CIDEr"""
        from bartphobeit.model import normalize_vietnamese_answer
        
        # Normalize predictions
        norm_predictions = [normalize_vietnamese_answer(pred) for pred in predictions]
        
        # Handle multi-GT: Nếu ground_truths là list of lists (all_correct_answers), dùng để tính max
        is_multi_gt = len(ground_truths) > 0 and isinstance(ground_truths[0], list)
        if is_multi_gt:
            norm_ground_truths = [[normalize_vietnamese_answer(gt) for gt in gts] for gts in ground_truths]
        else:
            norm_ground_truths = [normalize_vietnamese_answer(gt) for gt in ground_truths]
        
        metrics = {}
        
        # Basic accuracy metrics (max over GTs nếu multi)
        exact_matches = []
        for pred, gts in zip(norm_predictions, norm_ground_truths):
            if isinstance(gts, list):
                exact_matches.append(any(pred == gt for gt in gts))
            else:
                exact_matches.append(pred == gts)
        metrics['accuracy'] = np.mean(exact_matches)
        
        # Fuzzy matching with improved scoring (best over GTs)
        fuzzy_matches = []
        for pred, gts in zip(norm_predictions, norm_ground_truths):
            if isinstance(gts, list):
                best_fuzzy = max(
                    1.0 if pred == gt else 
                    0.8 if pred in gt or gt in pred else 
                    (len(set(pred.split()).intersection(set(gt.split()))) / len(set(pred.split()).union(set(gt.split()))) if set(pred.split()) and set(gt.split()) else 0.0)
                    for gt in gts
                )
                fuzzy_matches.append(best_fuzzy)
            else:
                if pred == gts:
                    fuzzy_matches.append(1.0)
                elif pred in gts or gts in pred:
                    fuzzy_matches.append(0.8)  # Partial credit
                else:
                    # Check word overlap
                    pred_words = set(pred.split())
                    gt_words = set(gts.split())
                    if pred_words and gt_words:
                        overlap = len(pred_words.intersection(gt_words))
                        total = len(pred_words.union(gt_words))
                        fuzzy_matches.append(overlap / total if total > 0 else 0.0)
                    else:
                        fuzzy_matches.append(0.0)
        metrics['fuzzy_accuracy'] = np.mean(fuzzy_matches)
        
        # Token-level F1 score (best over GTs)
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred, gts in zip(norm_predictions, norm_ground_truths):
            pred_tokens = set(pred.split())
            if isinstance(gts, list):
                best_f1 = best_precision = best_recall = 0.0
                for gt in gts:
                    gt_tokens = set(gt.split())
                    
                    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                        p = r = f = 1.0
                    elif len(pred_tokens) == 0:
                        p = r = f = 0.0
                    else:
                        common_tokens = pred_tokens.intersection(gt_tokens)
                        p = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                        r = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
                        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    
                    best_precision = max(best_precision, p)
                    best_recall = max(best_recall, r)
                    best_f1 = max(best_f1, f)
                
                precisions.append(best_precision)
                recalls.append(best_recall)
                f1_scores.append(best_f1)
            else:
                gt_tokens = set(gts.split())
                
                if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                    f1_scores.append(1.0)
                    precisions.append(1.0)
                    recalls.append(1.0)
                elif len(pred_tokens) == 0:
                    f1_scores.append(0.0)
                    precisions.append(0.0)
                    recalls.append(0.0)
                else:
                    common_tokens = pred_tokens.intersection(gt_tokens)
                    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
        
        metrics['precision'] = np.mean(precisions)
        metrics['recall'] = np.mean(recalls)
        metrics['f1_score'] = np.mean(f1_scores)
        
        # More robust BLEU and ROUGE scoring with better error handling (best over GTs)
        if BLEU_ROUGE_AVAILABLE and self.config.get('calculate_bleu_rouge', True):
            try:
                bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
                rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
                
                for pred, gts in zip(norm_predictions, norm_ground_truths):
                    pred_tokens = pred.split()
                    if isinstance(gts, list):
                        best_bleu1 = best_bleu2 = best_bleu3 = best_bleu4 = 0.0
                        best_r1 = best_r2 = best_rl = 0.0
                        for gt in gts:
                            gt_tokens = gt.split()
                            
                            # BLEU
                            for n in range(1, 5):
                                try:
                                    if len(gt_tokens) == 0:
                                        bleu = 0.0
                                    else:
                                        bleu = sentence_bleu([gt_tokens], pred_tokens, 
                                                             weights=tuple([1/n]*n + [0]*(4-n)),
                                                             smoothing_function=self.smoothing_function)
                                    if n == 1: best_bleu1 = max(best_bleu1, bleu)
                                    elif n == 2: best_bleu2 = max(best_bleu2, bleu)
                                    elif n == 3: best_bleu3 = max(best_bleu3, bleu)
                                    elif n == 4: best_bleu4 = max(best_bleu4, bleu)
                                except:
                                    pass
                            
                            # ROUGE
                            try:
                                rouge_result = self.rouge_scorer.score(gt, pred)
                                best_r1 = max(best_r1, rouge_result['rouge1'].fmeasure)
                                best_r2 = max(best_r2, rouge_result['rouge2'].fmeasure)
                                best_rl = max(best_rl, rouge_result['rougeL'].fmeasure)
                            except:
                                pass
                        
                        bleu_scores['bleu_1'].append(best_bleu1)
                        bleu_scores['bleu_2'].append(best_bleu2)
                        bleu_scores['bleu_3'].append(best_bleu3)
                        bleu_scores['bleu_4'].append(best_bleu4)
                        rouge_scores['rouge1'].append(best_r1)
                        rouge_scores['rouge2'].append(best_r2)
                        rouge_scores['rougeL'].append(best_rl)
                    else:
                        gt_tokens = gts.split()
                        
                        # BLEU (giữ nguyên cho single)
                        for n in range(1, 5):
                            try:
                                if len(gt_tokens) == 0:
                                    bleu_scores[f'bleu_{n}'].append(0.0)
                                else:
                                    bleu = sentence_bleu([gt_tokens], pred_tokens, 
                                                         weights=tuple([1/n]*n + [0]*(4-n)),
                                                         smoothing_function=self.smoothing_function)
                                    bleu_scores[f'bleu_{n}'].append(bleu)
                            except Exception as e:
                                bleu_scores[f'bleu_{n}'].append(0.0)
                        
                        # ROUGE (giữ nguyên)
                        try:
                            rouge_result = self.rouge_scorer.score(gts, pred)
                            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
                            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
                            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
                        except Exception as e:
                            rouge_scores['rouge1'].append(0.0)
                            rouge_scores['rouge2'].append(0.0)
                            rouge_scores['rougeL'].append(0.0)
                
                # Add averaged scores
                for key, scores in bleu_scores.items():
                    metrics[key] = np.mean(scores) if scores else 0.0
                    
                    zero_count = sum(1 for s in scores if s == 0.0)
                    metrics[f'{key}_zero_count'] = zero_count
                    metrics[f'{key}_nonzero_ratio'] = (len(scores) - zero_count) / len(scores) if scores else 0.0

                metrics['rouge1'] = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0
                metrics['rouge2'] = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0
                metrics['rougel'] = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
                
                for key in ['rouge1', 'rouge2', 'rougel']:
                    source_key = 'rougeL' if key == 'rougel' else key
                    if source_key in rouge_scores:
                        zero_count = sum(1 for s in rouge_scores[source_key] if s == 0.0)
                        metrics[f'{key}_zero_count'] = zero_count
                        metrics[f'{key}_nonzero_ratio'] = (len(rouge_scores[source_key]) - zero_count) / len(rouge_scores[source_key]) if rouge_scores[source_key] else 0.0
                
                print(f"✓ BLEU/ROUGE calculated successfully")
                
            except Exception as e:
                print(f"Error calculating BLEU/ROUGE: {e}")
                # Defaults (giữ nguyên)
        
        # Thêm CIDEr nếu config enable
        if self.config.get('calculate_cider', True):
            try:
                from pycocoevalcap.cider.cider import Cider
                cider_scorer = Cider()
                cider_scores = []
                for pred, gts in zip(norm_predictions, norm_ground_truths):
                    if isinstance(gts, list):
                        gts_norm = {i: [gt] for i, gt in enumerate(gts)}  # Format dict cho Cider
                        pred_dict = {0: [pred]}
                    else:
                        gts_norm = {0: [gts]}
                        pred_dict = {0: [pred]}
                    try:
                        score, _ = cider_scorer.compute_score(gts_norm, pred_dict)
                        cider_scores.append(score)
                    except Exception as e:
                        print(f"CIDEr error for sample: {e}")
                        cider_scores.append(0.0)
                metrics['cider'] = np.mean(cider_scores) if cider_scores else 0.0
                print(f"CIDEr: {metrics['cider']:.4f}")
            except ImportError:
                print("CIDEr not available. Install pycocoevalcap")
            except Exception as e:
                print(f"Error calculating CIDEr: {e}")
                metrics['cider'] = 0.0
        
        # Counts (giữ nguyên)
        metrics['exact_match_count'] = sum(exact_matches)
        metrics['total_count'] = len(predictions)

        # [ADDED] Tính thêm WUPS bằng evaluator
        try:
            from bartphobeit.evaluate import ComprehensiveVQAEvaluator
            evaluator = ComprehensiveVQAEvaluator(self.model.decoder_tokenizer, self.config)
            wups_metrics = evaluator.calculate_comprehensive_metrics(predictions, ground_truths)
            metrics['wups_0.0'] = wups_metrics.get('wups_0.0', 0.0)
            metrics['wups_0.9'] = wups_metrics.get('wups_0.9', 0.0)
            print(f"WUPS-0.0: {metrics['wups_0.0']:.4f} | WUPS-0.9: {metrics['wups_0.9']:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute WUPS ({e})")
            metrics['wups_0.0'] = 0.0
            metrics['wups_0.9'] = 0.0
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Enhanced checkpoint saving"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'current_stage': self.current_stage
        }
        
        # Save regular checkpoint
        # checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_the_lastest_epoch.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = 'best_vqa_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved: {best_path}")
        
        # Keep only last N checkpoints
        # self.cleanup_checkpoints()
        
        return checkpoint_path
    
    def cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        keep_n = self.config.get('keep_last_n_checkpoints', 1)
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_the_lastest_epoch.pth'))
        
        if len(checkpoints) > keep_n:
            # Sort by creation time
            checkpoints.sort(key=os.path.getctime)
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_n]:
                try:
                    os.remove(checkpoint)
                    print(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
                except:
                    pass

    def save_predictions(self, predictions, ground_truths, epoch, metrics, questions):
        """Save predictions and ground truths for analysis"""
        if not self.config.get('save_predictions', True):
            return
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config,
            'results': []
        }

        for i, (pred, gt, ques) in enumerate(zip(predictions[:500], ground_truths[:500], questions[:500])):  # Save first 500
            results['results'].append({
                'index': i,
                'question': ques,
                'prediction': pred,
                'ground_truth': gt,
                'normalized_prediction': normalize_vietnamese_answer(pred),
                'normalized_ground_truth': normalize_vietnamese_answer(gt)
            })
        
        # Save to JSON
        # results_file = f'predictions_epoch_{epoch+1}_{timestamp}.json'
        results_file = f'predictions_epoch.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Predictions saved to: {results_file}")
    
    def train(self, num_epochs):
        """Enhanced full training loop"""
        print(f"Starting enhanced training for {num_epochs} epochs:")
        print(f"  Stage 1 (Frozen encoders): epochs 1-{self.config['stage1_epochs']}")
        print(f"  Stage 2 (Partial unfreeze): epochs {self.config['stage1_epochs']+1}-{10}")

        try:
            with open('/root/modeltuner/predictions_epoch.json', 'r', encoding='utf-8') as f:
                previous_data = json.load(f)
                current_epoch = previous_data.get('epoch', -1)
        except:
            current_epoch = -1
        print(f"Continue fine-tune with epoch {current_epoch + 1}")

        for epoch in range(current_epoch + 1, current_epoch + num_epochs + 1):
            print(f"\nEpoch {epoch + 1}/{num_epochs + current_epoch + 1}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics, predictions, ground_truths, questions = self.evaluate_vqa()
            print('=' * 50, 'GROUND TRUTHS', '=' * 50)
            print(ground_truths)
            print('=' * 50, 'GROUND TRUTHS', '=' * 50)
            print('=' * 50, 'PREDICTIONS', '=' * 50)
            print(predictions)
            print('=' * 50, 'PREDICTIONS', '=' * 50)
            print('=' * 50, 'QUESTIONS', '=' * 50)
            print(questions)
            print('=' * 50, 'QUESTIONS', '=' * 50)
            
            # Print comprehensive results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Validation Metrics:")
            print(f"    Exact Match Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"    Fuzzy Accuracy: {val_metrics['fuzzy_accuracy']:.4f}")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall: {val_metrics['recall']:.4f}")
            print(f"    F1 Score: {val_metrics['f1_score']:.4f}")
            print(f"    CIDEr: {val_metrics.get('cider', 0.0):.4f}")
            print(f"    WUPS-0.0: {val_metrics.get('wups_0.0', 0.0):.4f}")
            print(f"    WUPS-0.9: {val_metrics.get('wups_0.9', 0.0):.4f}")
            
            # Safe access to BLEU/ROUGE metrics with default values
            if val_metrics.get('bleu_1') is not None:
                print(f"    BLEU-1: {val_metrics.get('bleu_1', 0.0):.4f}")
                print(f"    BLEU-4: {val_metrics.get('bleu_4', 0.0):.4f}")
            else:
                print(f"    BLEU-1: N/A (NLTK not available)")
                print(f"    BLEU-4: N/A (NLTK not available)")
                
            if val_metrics.get('rougel') is not None:
                print(f"    ROUGE-L: {val_metrics.get('rougel', 0.0):.4f}")
            elif val_metrics.get('rouge_l') is not None:
                print(f"    ROUGE-L: {val_metrics.get('rouge_l', 0.0):.4f}")
            else:
                print(f"    ROUGE-L: N/A (rouge-score not available)")
            
            # Diagnostic information for debugging
            print(f"    Additional Diagnostics:")
            if 'exact_match_count' in val_metrics:
                print(f"      Exact matches: {val_metrics['exact_match_count']}/{val_metrics['total_count']}")
            
            # Show sample predictions for debugging
            if predictions and len(predictions) > 0:
                print(f"    Sample Predictions (first 3):")
                for i in range(min(3, len(predictions))):
                    pred_norm = normalize_vietnamese_answer(predictions[i])
                    gt_norm = normalize_vietnamese_answer(ground_truths[i])
                    match_status = "✓" if pred_norm == gt_norm else "✗"
                    print(f"      {match_status} Pred: '{pred_norm}' | GT: '{gt_norm}'")
            
            # Log to wandb with safe key access
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_fuzzy_accuracy': val_metrics['fuzzy_accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1_score': val_metrics['f1_score']
                }
                
                # Safely add BLEU/ROUGE metrics if available
                if val_metrics.get('bleu_1') is not None:
                    log_dict.update({
                        'val_bleu_1': val_metrics.get('bleu_1', 0.0),
                        'val_bleu_2': val_metrics.get('bleu_2', 0.0),
                        'val_bleu_3': val_metrics.get('bleu_3', 0.0),
                        'val_bleu_4': val_metrics.get('bleu_4', 0.0)
                    })
                
                if val_metrics.get('rougel') is not None:
                    log_dict.update({
                        'val_rouge1': val_metrics.get('rouge1', 0.0),
                        'val_rouge2': val_metrics.get('rouge2', 0.0),
                        'val_rougel': val_metrics.get('rougel', 0.0)
                    })
                elif val_metrics.get('rouge_l') is not None:
                    log_dict.update({
                        'val_rouge_l': val_metrics.get('rouge_l', 0.0)
                    })
                
                wandb.log(log_dict)
            
            # Check if this is the best model
            is_best = val_metrics['fuzzy_accuracy'] > self.best_fuzzy_accuracy
            if is_best:
                self.best_fuzzy_accuracy = val_metrics['fuzzy_accuracy']
                self.best_accuracy = val_metrics['accuracy']
                self.best_f1 = val_metrics['f1_score']
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every_n_epochs', 1) == 0:
                checkpoint_path = self.save_checkpoint(epoch, val_metrics, is_best)
                print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
            
            # Save predictions
            self.save_predictions(predictions, ground_truths, epoch, val_metrics, questions)
            
            if is_best:
                print(f"🎉 New best fuzzy accuracy: {self.best_fuzzy_accuracy:.4f}")
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Best Results:")
        print(f"  Fuzzy Accuracy: {self.best_fuzzy_accuracy:.4f}")
        print(f"  Exact Match Accuracy: {self.best_accuracy:.4f}")
        print(f"  F1 Score: {self.best_f1:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'final_best_fuzzy_accuracy': self.best_fuzzy_accuracy,
                'final_best_accuracy': self.best_accuracy,
                'final_best_f1': self.best_f1
            })
            wandb.finish()
        
        # [ADDED] Lưu final model
        # final_path = 'final_vqa_model.pth'
        # torch.save(self.model.state_dict(), final_path)
        # print(f"✓ Final trained model saved to {final_path}")
        
        return self.best_fuzzy_accuracy

    def train_epoch(self, epoch):
        """Enhanced training epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Stage management
        if epoch == self.config['stage1_epochs']:
            print("\n" + "="*60)
            print("SWITCHING TO STAGE 2: Partial encoder unfreezing")
            print("="*60)
            self.model.partial_unfreeze(self.config['unfreeze_last_n_layers'])
            self.setup_optimizers_and_schedulers()  # Recreate optimizer
            remaining_steps = len(self.train_loader) * (self.config['num_epochs'] - epoch)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,  # No warmup sau unfreeze
                num_training_steps=remaining_steps
            )
            print(f"Adjusted scheduler for remaining {remaining_steps} steps")
            self.current_stage = 2
        
        progress_bar = tqdm(self.train_loader, desc=f"Stage {self.current_stage} - Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass với FP16
            ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if self.scaler else nullcontext()
            with ctx:
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    question_input_ids=batch['question_input_ids'],
                    question_attention_mask=batch['question_attention_mask'],
                    answer_input_ids=batch['answer_input_ids'],
                    answer_attention_mask=batch['answer_attention_mask']
                )
            
            loss = outputs.loss / self.accumulation_steps  # Scale loss cho accumulation
            
            # Backward pass với scaler
            if self.scaler:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Scheduler step mỗi accumulation steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps  # Adjust avg
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}",
                'LR': f"{current_lr:.2e}",
                'Stage': self.current_stage
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'stage': self.current_stage,
                    'global_step': self.global_step
                })
            
            # More frequent evaluation during early training for debugging
            eval_freq = self.config.get('evaluate_every_n_steps', 1000)
            if epoch < 2:  # More frequent eval in first 2 epochs
                eval_freq = min(eval_freq, 5000)
            
            if (eval_freq > 0 and self.global_step % eval_freq == 0):
                print(f"\nEvaluating at step {self.global_step}...")
                val_metrics, predictions, ground_truths, questions = self.evaluate_vqa()
                
                # Step-level logging for debugging
                print(f"Step {self.global_step} metrics:")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Fuzzy Accuracy: {val_metrics['fuzzy_accuracy']:.4f}")
                print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
                
                # BLEU/ROUGE diagnostic logging
                if val_metrics.get('bleu_1') is not None:
                    print(f"  BLEU-1: {val_metrics.get('bleu_1', 0.0):.4f} (zero count: {val_metrics.get('bleu_zero_count', 'N/A')})")
                if val_metrics.get('rougel') is not None or val_metrics.get('rouge_l') is not None:
                    rouge_score = val_metrics.get('rougel', val_metrics.get('rouge_l', 0.0))
                    print(f"  ROUGE-L: {rouge_score:.4f}")
                
                if self.use_wandb:
                    step_log_dict = {
                        'step_val_accuracy': val_metrics['accuracy'],
                        'step_val_fuzzy_accuracy': val_metrics['fuzzy_accuracy'],
                        'step_val_f1_score': val_metrics['f1_score'],
                        'global_step': self.global_step
                    }
                    
                    # Safe BLEU/ROUGE logging
                    if val_metrics.get('bleu_1') is not None:
                        step_log_dict['step_val_bleu_1'] = val_metrics.get('bleu_1', 0)
                    if val_metrics.get('rougel') is not None:
                        step_log_dict['step_val_rougel'] = val_metrics.get('rougel', 0)
                    elif val_metrics.get('rouge_l') is not None:
                        step_log_dict['step_val_rouge_l'] = val_metrics.get('rouge_l', 0)
                    
                    wandb.log(step_log_dict)
                
                self.model.train()  # Back to training mode
        
        return total_loss / num_batches