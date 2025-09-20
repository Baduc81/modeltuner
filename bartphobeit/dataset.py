from torch.utils.data import Dataset
from bartphobeit.BARTphoBEIT import VietnameseVQADataset
import random
from bartphobeit.model import augment_question
import os  # THÊM: Import global để tránh inline
from PIL import Image
import torch
from torchvision import transforms
import torch

class EnhancedVietnameseVQADataset(VietnameseVQADataset):
    """Enhanced dataset with data augmentation and multiple correct answers support"""
    
    def __init__(self, questions, image_dir, question_tokenizer, answer_tokenizer, 
                 feature_extractor, max_length=128, transform=None, use_augmentation=False, 
                 augment_ratio=0.2, use_multiple_answers=True):
        super().__init__(questions, image_dir, question_tokenizer, answer_tokenizer, 
                        feature_extractor, max_length, transform)
        self.use_augmentation = use_augmentation
        self.augment_ratio = augment_ratio
        self.use_multiple_answers = use_multiple_answers
        self.is_training = True
        
        # THÊM: Tăng max_length answer cho BARTpho-syllable (seq dài hơn word, theo paper [web:1,8])
        self.answer_max_length = 64  # Từ 32
        
        if use_augmentation:
            print(f"Data augmentation enabled with ratio: {augment_ratio}")
        if use_multiple_answers:
            print(f"Multiple correct answers support enabled")
    
    def set_training(self, training=True):
        """Set training mode for data augmentation"""
        self.is_training = training
    
    # THAY ĐỔI: Remove __getitem__ đầu (bug super call với question_data param; parent chỉ nhận idx)
    # Giữ chỉ override full __getitem__ (handle modified data)

    def __getitem__(self, idx, question_data=None):
        """Override to handle custom question_data with multiple answers"""
        if question_data is None:
            question_data = self.questions[idx].copy()  # THAY ĐỔI: Copy để modify safe
        
        # THÊM: Apply data augmentation only during training (di chuyển từ __getitem__ đầu)
        if self.use_augmentation and self.is_training:
            question_data['question'] = augment_question(
                question_data['question'], self.augment_ratio
            )
        
        # Nếu training và có nhiều đáp án → chọn random
        if self.use_multiple_answers and self.is_training and len(question_data.get('answers', [])) > 1:
            question_data['ground_truth'] = random.choice(question_data['answers'])
        else:
            # Validation/test → giữ ground_truth gốc hoặc lấy answer[0]
            if 'ground_truth' not in question_data and 'answers' in question_data:
                question_data['ground_truth'] = question_data['answers'][0]
        
        # Load and process image (giữ, nhưng import global)
        image_path = os.path.join(self.image_dir, question_data['image_name'] + '.jpg')
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Create a default image if loading fails
            image = Image.new('RGB', (224, 224), color='black')  # Black neutral
        
        # Process image with ViT feature extractor
        image_inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Tokenize question with question tokenizer (PhoBERT)
        question = question_data['question']
        question_encoding = self.question_tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Use selected answer (or first answer if no selection made)
        answer = question_data['ground_truth']
        answer_encoding = self.answer_tokenizer(
            answer,
            max_length=self.answer_max_length,  # THÊM: Tăng cho syllable
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'question_input_ids': question_encoding['input_ids'].squeeze(0),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(0),
            'answer_input_ids': answer_encoding['input_ids'].squeeze(0),
            'answer_attention_mask': answer_encoding['attention_mask'].squeeze(0),
            'question_text': question,  # Original question text
            'answer_text': answer,  # Current selected answer (single for train)
            'all_correct_answers': question_data.get('all_correct_answers', [answer])  # All GTs for eval (multi support)
        }