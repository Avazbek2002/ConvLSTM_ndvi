import torch
import torch.nn as nn
import torch.optim as optim
from data.NDVI_dataset import NDVIDataset
from model.model import Encoder, Decoder, Seq2Seq
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os

if __name__ == '__main__':
    # --- Hyperparameters ---
    # !! IMPORTANT: Update this path to your dataset !!
    DATAROOT = "/Users/isroilov/Desktop/Pasture/patched" 

    INPUT_CHANNELS = 2
    HIDDEN_CHANNELS = 32
    KERNEL_SIZE = (3, 3)
    
    # Data parameters from your Dataset class
    INPUT_SEQ_LENGTH = 4
    FUTURE_SEQ_LENGTH = 1
    
    # Training parameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    TEACHER_FORCING_RATIO = 0.5
    MODEL_SAVE_PATH = 'convlstm_seq2seq_ndvi.pth'

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    
    if not os.path.exists(DATAROOT):
        print(f"Error: Data directory not found at '{DATAROOT}'")
        print("Please update the DATAROOT variable in the script.")
        exit()

    # --- Dataset and DataLoader ---
    print("Loading dataset...")
    train_dataset = NDVIDataset(dataroot=DATAROOT, 
                                input_length=INPUT_SEQ_LENGTH, 
                                output_length=FUTURE_SEQ_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded with {len(train_dataset)} samples.")

    # --- Model, Optimizer, and Loss ---
    encoder = Encoder(INPUT_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE).to(device)
    decoder = Decoder(INPUT_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    print("--- Starting Training ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for i, batch in enumerate(progress_bar):
            # The dataloader gives (b, t, c, h, w)
            # We need (t, b, c, h, w) for the model
            if i > 10:
                break

            input_tensor = batch['input'].permute(1, 0, 2, 3, 4).to(device)
            target_tensor = batch['output'].permute(1, 0, 2, 3, 4).to(device)
            
            # --- Training Step ---
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(input_tensor, target_tensor, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            
            # Calculate loss
            loss = criterion(predictions, target_tensor)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            print(loss.item())
            
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.6f}")

    print("--- Finished Training ---")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
