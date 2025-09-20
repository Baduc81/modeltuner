import torch

def get_improved_config():
    """Improved configuration with BARTPhoBEiT enhancements"""
    return {
        'vision_model': 'google/vit-large-patch16-224-in21k',  # Thay từ base
        'text_model': 'vinai/phobert-large',  # Giữ
        'decoder_model': 'vinai/bartpho-syllable',  # Thay từ word

        'hidden_dim': 1024,  # PhoBERT-large dimension
        'max_length': 128,
        # 'batch_size': 32,
        'batch_size': 2,
        'num_epochs': 20,
        'image_dir': '/root/modeltuner/data/preprocessed_images',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # BARTPhoBEiT specific configurations
        'use_vqkd': True,  # Enable VQ-KD Visual Tokenizer
        'visual_vocab_size': 8192,  # Codebook size
        'num_multiway_layers': 8,  # Number of Multiway Transformer layers
        
        # Unified Masked Data Modeling
        'use_unified_masking': True,
        'text_mask_ratio': 0.15,  # 15% for monomodal text
        'multimodal_text_mask_ratio': 0.50,  # 50% for multimodal text
        'vision_mask_ratio': 0.40,  # 40% for image patches
        
        # Staged training configuration
        'stage1_epochs': 5,  # Freeze encoders
        'stage2_epochs':15,  # Partial unfreeze
        
        # Different learning rates
        'decoder_lr': 1e-4,
        'encoder_lr': 1e-5,
        'vision_lr': 5e-6,
        
        # Enhanced scheduler configuration
        'warmup_ratio': 0.1,
        'scheduler_type': 'linear_decay_with_warmup',
        'weight_decay': 0.01,
        
        # Enhanced regularization
        'label_smoothing': 0.1,
        'dropout_rate': 0.2,
        
        # Unfreezing strategy
        'unfreeze_last_n_layers': 6,
        
        # Data augmentation
        'use_data_augmentation': True,
        'augment_ratio': 0.2,
        
        # Logging and checkpoints
        'use_wandb': False,
        'project_name': 'BARTPhoBEiT-Vietnamese-VQA',
        'save_every_n_epochs': 1,
        'keep_last_n_checkpoints': 1,
        
        # Enhanced evaluation
        'evaluate_every_n_steps': 5000,
        'save_predictions': True,
        'calculate_bleu_rouge': True,
        'calculate_cider': True,

        'use_fp16': True,  # Sử dụng mixed precision training
        'gradient_accumulation_steps': 2,  # Tăng kích thước batch hiệu quả
    }