import torch
from torch.utils.data import DataLoader
from dataset import NuScenesDataset
from models import Model, FlatCNN
from unet_model import UNet
import matplotlib.pyplot as plt
import time
import wandb

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

def KLDivLoss(logits, target):
    # Compute log-softmax of logits for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # Convert target to probabilities
    target_probs = torch.nn.functional.softmax(target, dim=1)

    # Compute KL divergence loss
    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='batchmean')

    return loss

def masked_KLDivLoss(logits, target, mask):
    # Compute log-softmax of logits for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # Convert target to probabilities
    target_probs = torch.nn.functional.softmax(target, dim=1)

    # Compute KL divergence loss
    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='none')

    # Apply mask
    loss = loss * mask

    # Compute mean over batch and spatial dimensions
    loss = loss.mean()

    return loss

def train():
    # Config
    batchSize = 10
    lr = 1e-5
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
    loss_function = masked_KLDivLoss

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

            # Compute the output free map as 1 - output_static - output_dynamic
            target_free = 1 - sample_batched['output_static'] - sample_batched['output_dynamic']

            # Concatenate the output static, dynamic and free maps along the channel dimension
            target = torch.cat((sample_batched['output_static'], sample_batched['output_dynamic'], target_free), dim=1)
            
            # Convert target to class labels  -  NEED TO BE REMOVED
            #target = target.argmax(dim=1)

            mask = compute_mask(sample_batched['output_instant'])
            loss = loss_function(output, target, mask)
            #loss = loss_function(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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