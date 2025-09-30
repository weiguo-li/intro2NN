from transformers.models.qwen3 import modeling_qwen3
from modeling_qwen_attention import Qwen3Attention_v1
modeling_qwen3.Qwen3Attention = Qwen3Attention_v1
from reward_model import OutcomeRewardModel
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

dataset = None

# Load Qwen3 tokenizer and model
model_name = "Qwen/Qwen3-0.6B"
model_path = "/home/students/wli/UniHeidelberg/semster2/final_projects/models/Qwen3-0.6B-Base"
# model_path = model_name
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", attn_implementation="eager")  # Specify eager for attention before softmax

# tokenizer.padding_side = "left"
# Make sure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

orm = OutcomeRewardModel(model)

from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import gc

# Configuration
epoch = 1
eval_step = 1000
batch_size = 8
learning_rate = 1e-5
max_grad_norm = 1.0
gradient_accumulation_steps = 1  # Increase if you still have memory issues

# Setup optimizer
optimizer = AdamW(orm.parameters(), lr=learning_rate)

# Enable mixed precision training for memory efficiency
scaler = torch.amp.GradScaler('cuda')

def collate_fn(examples):
    """Optimized collate function"""
    chosen_input = tokenizer(
        [ex['chosen'] for ex in examples], 
        return_tensors='pt', 
        padding=True,
        truncation=True,  # Add truncation to limit sequence length
        max_length=512  # Adjust based on your needs
    )
    rejected_input = tokenizer(
        [ex['rejected'] for ex in examples], 
        return_tensors='pt', 
        padding=True,
        truncation=True,
        max_length=512
    )
    return chosen_input, rejected_input

# Training loop
step = 0
orm.train()  # Ensure model is in training mode


# filter ds so that the length of chosen and rejected is less than 512
ds['train'] = ds['train'].filter(lambda x: len(tokenizer(x['chosen'])['input_ids']) < 196 and len(tokenizer(x['rejected'])['input_ids']) < 196, num_proc=4)


for epoch_idx in range(epoch):
    dataloader = DataLoader(
        ds['train'], 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        pin_memory=True,  # Faster data transfer to GPU
        num_workers=2  # Parallel data loading
    )
    
    for batch_idx, (chosen_input, rejected_input) in enumerate(tqdm(dataloader, leave=True)):
        # Move inputs to device
        chosen_input = {k: v.to(orm.device) for k, v in chosen_input.items()}
        rejected_input = {k: v.to(orm.device) for k, v in rejected_input.items()}

        
        # Use mixed precision training
        with torch.amp.autocast('cuda'):
            # Forward pass
            reward_chosen = orm(**chosen_input)
            reward_rejected = orm(**rejected_input)
            
            # Calculate loss
            diff = reward_chosen - reward_rejected
            loss = torch.nn.functional.softplus(-diff).mean()
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights only after accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(orm.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            step += 1
            
            # Evaluation logging
            if step % eval_step == 0:
                print(f"Step {step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        # Critical: Delete tensors and clear cache periodically
        del chosen_input, rejected_input, reward_chosen, reward_rejected, diff, loss
        
        if batch_idx % 50 == 0:  # Clear cache every 50 batches
            torch.cuda.empty_cache()
            gc.collect()

# Save model
import os

save_path = "../reward_model.pt"
save_dir = os.path.dirname(save_path)

# Create directory if it doesn't exist
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


torch.save(orm.state_dict(), save_path)
print("Training completed and model saved!")

# Final cleanup
torch.cuda.empty_cache()
gc.collect()


def main():
    print("Hello from intro2nn!")


if __name__ == "__main__":
    main()
