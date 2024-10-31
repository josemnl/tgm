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

def loadAndArrangeSample(sample_batched, device):
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

    # Compute the output free map as 1 - output_static - output_dynamic
    target_free = 1 - sample_batched['output_static'] - sample_batched['output_dynamic']

    # Concatenate the output static, dynamic and free maps along the channel dimension
    target = torch.cat((sample_batched['output_static'], sample_batched['output_dynamic'], target_free), dim=1)

    mask = compute_mask(sample_batched['output_instant'])

    return input, target, mask

def train():
    # Config
    batchSize = 10
    lr = 1e-5
    epochs = 10
    modelType = 'UNet'
    val_periods = 100
    val_batches = 10

    # Initialize wandb
    wandb.init(project="TGM", name=modelType + "_batchSize_" + str(batchSize) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_date_" + time.strftime("%Y%m%d-%H%M%S"),
               config={
        "batchSize": batchSize,
        "lr": lr,
        "epochs": epochs
    })

    # Load train and validation datasets
    train_dataset = NuScenesDataset(mode='train')
    val_dataset = NuScenesDataset(mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)

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
        for i_batch, sample_batched in enumerate(train_dataloader):
            # Load and arrange sample
            input, target, mask = loadAndArrangeSample(sample_batched, device)

            # Forward pass
            output = model(input)

            # Compute loss
            loss = loss_function(output, target, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            print(f"Epoch {epoch}, Batch {i_batch}/{len(train_dataloader)}, Loss: {loss.item()}")

            # Log loss
            wandb.log({"loss": loss.item()})

            # Print time
            print(f"Time: {time.time() - time_prev}")
            print("Time per batch: ", (time.time() - time_prev) / batchSize)
            print("Average time per sample: ", (time.time() - time_start) / ((i_batch + 1) * batchSize)) # I NEED TO FIX THIS
            print('')
            time_prev = time.time()

            # Validation
            if i_batch % val_periods == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_sum = 0
                    for i_batch_val, sample_batched_val in enumerate(val_dataloader):
                        if i_batch_val >= val_batches:
                            break
                        # Load and arrange sample
                        input_val, target_val, mask_val = loadAndArrangeSample(sample_batched_val, device)

                        # Forward pass
                        output_val = model(input_val)

                        # Compute loss
                        loss_val = loss_function(output_val, target_val, mask_val)

                        # Accumulate loss
                        val_loss_sum += loss_val.item()

                    # Compute average validation loss
                    avg_val_loss = val_loss_sum / val_batches

                    # Print average validation loss
                    print(f"Validation, Average Loss: {avg_val_loss}")

                    # Log average validation loss
                    wandb.log({"val_loss": avg_val_loss})

                model.train()

    # Save model
    torch.save(model.state_dict(), modelType + "_batchSize_" + str(batchSize) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_date_" + time.strftime("%Y%m%d-%H%M%S") + ".pt")

    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    train()