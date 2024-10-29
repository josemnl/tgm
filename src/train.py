import torch
from torch.utils.data import DataLoader
from dataset import NuScenesDataset
from models import Model, FlatCNN
from unet_model import UNet
import matplotlib.pyplot as plt
import time
import wandb

def masked_cross_entropy(logits, target, mask):
    """
    Calculates the cross-entropy loss with masking.

    Args:
        logits (torch.Tensor): Predicted logits (unnormalized probabilities) of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size).
        mask (torch.Tensor): Mask indicating valid positions, of shape (batch_size).

    Returns:
        torch.Tensor: The masked cross-entropy loss.
    """

    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, target)
    loss = loss * mask
    return loss.sum() / mask.sum()

def compute_mask(output_instant):
    """
    Computes the mask based on output_instant.
    """

    # Valid cells are those with a value > 0.7 or < 0.4
    mask = (output_instant > 0.7) | (output_instant < 0.4)

    # Plot output_instant
    #plt.imshow(output_instant[0, 0, :, :].detach().cpu().numpy())
    #plt.show()

    # Plot the mask
    #plt.imshow(mask[0, 0, :, :].detach().cpu().numpy())
    #plt.show()
    return mask

def train():
    # Config
    batchSize = 10
    lr = 1e-3
    epochs = 10
    modelType = 'UNet'

    # Initialize wandb
    wandb.init(project="TGM", name=modelType + "_batchSize_" + str(batchSize) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_date_" + time.strftime("%Y%m%d-%H%M%S"),
               config={
        "batchSize": batchSize,
        "lr": lr,
        "epochs": epochs
    })

    # Load dataset
    dataset = NuScenesDataset()
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load model
    if modelType == 'Model':
        model = Model().to(device)
    elif modelType == 'FlatCNN':
        model = FlatCNN().to(device)
    elif modelType == 'UNet':
        model = UNet(2,3).to(device)

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load loss function
    #loss_function = torch.nn.CrossEntropyLoss()
    loss_function = masked_cross_entropy

    time_prev = time.time()
    time_start = time_prev

    # Train
    for epoch in range(epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            # Move data to device
            sample_batched['input_static'] = sample_batched['input_static'].to(device)
            sample_batched['input_dynamic'] = sample_batched['input_dynamic'].to(device)
            sample_batched['output_static'] = sample_batched['output_static'].to(device)
            sample_batched['output_dynamic'] = sample_batched['output_dynamic'].to(device)
            sample_batched['output_instant'] = sample_batched['output_instant'].to(device)

            # Unsqueeze data to add the channel dimension
            sample_batched['input_static'] = sample_batched['input_static'].unsqueeze(1)
            sample_batched['input_dynamic'] = sample_batched['input_dynamic'].unsqueeze(1)
            sample_batched['output_static'] = sample_batched['output_static'].unsqueeze(1)
            sample_batched['output_dynamic'] = sample_batched['output_dynamic'].unsqueeze(1)
            sample_batched['output_instant'] = sample_batched['output_instant'].unsqueeze(1)

            # # Concatenate the input static and dynamic maps along the channel dimension
            input = torch.cat((sample_batched['input_static'], sample_batched['input_dynamic']), dim=1)

            # Forward pass
            output = model(input)

            # Apply softmax
            output = torch.nn.functional.softmax(output, dim=1)

            # Discard the free map
            output = output[:, 0:2, :, :]

            # Concatenate the output static and dynamic maps along the channel dimension
            target = torch.cat((sample_batched['output_static'], sample_batched['output_dynamic']), dim=1)

            # Compute loss
            mask = compute_mask(sample_batched['output_instant'])
            loss = loss_function(output, target, mask)

            # Zero gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()

            # Print loss
            print(f"Epoch {epoch}, Batch {i_batch}/{len(dataloader)}, Loss: {loss.item()}")

            # Log loss
            wandb.log({"loss": loss.item()})

            # Print time
            print(f"Time: {time.time() - time_prev}")
            print("Time per batch: ", (time.time() - time_prev) / batchSize)
            print("Average time per sample: ", (time.time() - time_start) / ((i_batch + 1) * batchSize))
            print('')
            time_prev = time.time()

if __name__ == "__main__":
    train()