import torch
from torch.utils.data import DataLoader
from dataset import NuScenesDataset
from models import Model, FlatCNN
from unet_model import UNet
import matplotlib.pyplot as plt
import time
import wandb
import yaml
import os

def compute_mask(output_instant, isKeyFrame):
    """
    Computes the mask based on output_instant.
    """

    # Valid cells are those with a value > 0.7 or < 0.4
    mask = (output_instant > 0.7) | (output_instant < 0.4)

    # Expand to have the same mask for all channels
    print('mask shape: ', mask.shape)
    mask = mask.expand(-1, 3, -1, -1).clone()
    print('mask shape: ', mask.shape)

    # For each sample, if isKeyFrame is True, the dynamic mask is all ones
    for i in range(mask.shape[0]):
        if isKeyFrame[i]:
            mask[i, 1, :, :] = 1
    
    return mask

def KLDivLoss(logits, target):
    # Compute log-softmax of logits for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # Convert target to probabilities
    target_probs = torch.nn.functional.softmax(target, dim=1)

    # Compute KL divergence loss
    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='batchmean')

    return loss

def biased_KLDivLoss(logits, target, dynamic_weight=10.0, static_weight=0.0):
    # Compute log-softmax of logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # Convert target to probabilities
    target_probs = torch.nn.functional.softmax(target, dim=1)

    # Compute KL divergence loss without reduction
    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='none')

    # Identify dynamic and static cells in the target as those with a probability > 0.5
    dynamic_cells = target_probs[:, 1, :, :] > 0.5
    static_cells = target_probs[:, 0, :, :] > 0.5

    # Expand to match loss dimensions
    dynamic_cells = dynamic_cells.unsqueeze(1).expand_as(loss)
    static_cells = static_cells.unsqueeze(1).expand_as(loss)

    # Create a weighting mask
    weights = torch.ones_like(loss)
    weights[dynamic_cells] = dynamic_weight
    weights[static_cells] = static_weight

    # Apply weights to the loss
    loss = loss * weights

    # Compute mean loss
    loss = loss.mean()

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

def masked_biased_KLDivLoss(logits, target, mask, dynamic_weight=1.0, static_weight=1.0, free_weight=1.0):
    # Compute log-softmax of logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # Make sure the target probabilities sum to 1
    epsilon = 1e-8
    target_probs = target / (target.sum(dim=1, keepdim=True) + epsilon)
    target_probs = torch.clamp(target_probs, min=epsilon, max=1.0)

    # Compute KL divergence loss without reduction
    loss = torch.nn.functional.kl_div(log_probs, target_probs, reduction='none')

    # Apply mask
    loss = loss * mask

    # Apply dynamic, static and free weights
    loss[:, 0, :, :] = loss[:, 0, :, :] * static_weight
    loss[:, 1, :, :] = loss[:, 1, :, :] * dynamic_weight
    loss[:, 2, :, :] = loss[:, 2, :, :] * free_weight

    # Compute mean loss
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

    mask = compute_mask(sample_batched['output_instant'], sample_batched['isKeyFrame'])

    return input, target, mask

def plot(input, target, output, mask):
    # Transform output to probabilities
    output_prob = torch.nn.functional.softmax(output, dim=1)
    # Compute grayscale as 1 - transpose
    input_image = 1 - torch.transpose(input, 2, 3)
    target_image = 1 - torch.transpose(target, 2, 3)
    output_image = 1 - torch.transpose(output_prob, 2, 3)
    mask_image = torch.transpose(mask, 2, 3)
    # Plot input, target, output and mask for the static and dynamic maps
    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(input_image[0, 0, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0, 0].set_title('Input Static')
    axs[1, 0].imshow(input_image[0, 1, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1, 0].set_title('Input Dynamic')
    axs[0, 1].imshow(target_image[0, 0, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0, 1].set_title('Target Static')
    axs[1, 1].imshow(target_image[0, 1, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1, 1].set_title('Target Dynamic')
    axs[0, 2].imshow(output_image[0, 0, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0, 2].set_title('Output Static')
    axs[1, 2].imshow(output_image[0, 1, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1, 2].set_title('Output Dynamic')
    axs[0, 3].imshow(mask_image[0, 0, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0, 3].set_title('Mask Static')
    axs[1, 3].imshow(mask_image[0, 1, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1, 3].set_title('Mask Dynamic')

    # Remove axis
    for ax in axs.flatten():
        ax.axis('off')

    return fig

def loadConfig(filename):
    with open(filename) as file:
        config = yaml.safe_load(file)
    #config = SimpleNamespace(**config)
    return config

def train():
    # Config
    conf = loadConfig('./trainConfig/train.yaml')

    name = conf['modelType'] + "_batchSize_" + str(conf['batchSize']) + "_lr_" + str(conf['lr']) + "_epochs_" + str(conf['epochs']) + "_date_" + time.strftime("%Y%m%d-%H%M%S")

    # Create a folder to save the config and models
    if not os.path.exists('./trainRuns/' + name):
        os.makedirs('./trainRuns/' + name)

    # Save config
    with open('./trainRuns/' + name + '/config.yaml', 'w') as file:
        yaml.dump(conf, file)

    # Initialize wandb
    if conf['isWandb']:
        wandb.init(project="TGM", name=name, config=conf)

    # Load train and validation datasets
    train_dataset = NuScenesDataset(mode='train', isAugment=conf['isAugment'], isLabeledTraining=conf['isLabeledTraining'])
    val_dataset = NuScenesDataset(mode='val', isAugment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=conf['batchSize'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf['batchSize'], shuffle=True)

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load model
    if conf['modelType'] == 'Model':
        model = Model().to(device)
    elif conf['modelType'] == 'FlatCNN':
        model = FlatCNN().to(device)
    elif conf['modelType'] == 'UNet':
        model = UNet(2,3).to(device)

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])

    # Load loss function
    assert conf['lossFunction'] in ['KLDivLoss', 'masked_KLDivLoss', 'biased_KLDivLoss', 'masked_biased_KLDivLoss']

    if conf['lossFunction'] == 'KLDivLoss':
        loss_function = KLDivLoss
    elif conf['lossFunction'] == 'masked_KLDivLoss':
        loss_function = masked_KLDivLoss
    elif conf['lossFunction'] == 'biased_KLDivLoss':
        loss_function = biased_KLDivLoss
    elif conf['lossFunction'] == 'masked_biased_KLDivLoss':
        loss_function = masked_biased_KLDivLoss

    time_prev = time.time()
    time_start = time_prev

    # Train
    for epoch in range(conf['epochs']):
        for i_batch, sample_batched in enumerate(train_dataloader):
            # Load and arrange sample
            input, target, mask = loadAndArrangeSample(sample_batched, device)

            # Forward pass
            output = model(input)

            # Compute loss
            if conf['lossFunction'] == 'masked_KLDivLoss' or conf['lossFunction'] == 'masked_biased_KLDivLoss':
                loss = loss_function(output, target, mask)
            else:
                loss = loss_function(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss and times
            print(f"Epoch {epoch}, Batch {i_batch}/{len(train_dataloader)}, Loss: {loss.item()}")
            print(f"Time for this batch: {time.time() - time_prev}")
            print("Average time per batch: ", (time.time() - time_start) / (epoch * len(train_dataloader) + i_batch + 1))
            print('')
            time_prev = time.time()

            # Log loss
            if conf['isWandb']:
                wandb.log({"loss": loss.item()})

            # Validation
            if i_batch % conf['val_periods'] == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_sum = 0
                    for i_batch_val, sample_batched_val in enumerate(val_dataloader):
                        if i_batch_val >= conf['val_batches']:
                            break
                        # Load and arrange sample
                        input_val, target_val, mask_val = loadAndArrangeSample(sample_batched_val, device)

                        # Forward pass
                        output_val = model(input_val)

                        # Compute loss
                        if conf['lossFunction'] == 'masked_KLDivLoss' or conf['lossFunction'] == 'masked_biased_KLDivLoss':
                            loss_val = loss_function(output_val, target_val, mask_val)
                        else:
                            loss_val = loss_function(output_val, target_val)

                        # Accumulate loss
                        val_loss_sum += loss_val.item()

                    # Compute average validation loss
                    avg_val_loss = val_loss_sum / conf['val_batches']

                    # Print average validation loss
                    print(f"Validation, Average Loss: {avg_val_loss}")

                    # Plot input, target, output and mask
                    fig = plot(input, target, output, mask)

                    # Log average validation loss and plot
                    if conf['isWandb']:
                        wandb.log({"val_loss": avg_val_loss, "plot": fig})

                model.train()
            
            # Save checkpoint
            if i_batch % conf['savePeriods'] == 0:
                torch.save(model.state_dict(), './trainRuns/' + name + '/checkpoint' + str(epoch) + '_' + str(i_batch) + '.pt')

    # Save model
    torch.save(model.state_dict(), './trainRuns/' + name + '/model.pt')

    # Close wandb
    if conf['isWandb']:
        wandb.finish()

if __name__ == "__main__":
    train()