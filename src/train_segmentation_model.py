# Importing necessary libraries
import torch  # PyTorch for building and training neural networks
import torch.nn as nn  # Provides building blocks for neural networks, like layers
import torch.optim as optim  # Optimization algorithms like Adam for training
from torch.utils.data import Dataset, DataLoader  # For loading and batching datasets
from torchvision import transforms  # Common image transformations for preprocessing
from PIL import Image  # For image input/output
import os  # For interacting with directories and file paths

# Define the Encoder class, which compresses an image into a latent vector.
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):  # Constructor, where latent_dim is the size of the compressed latent vector
        super(Encoder, self).__init__()  # Initialize the parent class (nn.Module)
        # Define 3 convolutional layers with increasing filter sizes to extract features from the input image.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # Input: RGB image (3 channels), Output: 64 feature maps
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Output: 128 feature maps
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # Output: 256 feature maps
        # Fully connected layer to compress the output of the convolutional layers into a latent vector of size latent_dim
        self.fc = nn.Linear(256 * 16 * 16, latent_dim)  # Image size is assumed to be downsampled to 16x16 by the conv layers

    def forward(self, x):
        # Apply ReLU activation after each convolution to introduce non-linearity
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Flatten the feature maps into a vector
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, num_features)
        x = self.fc(x)  # Pass through the fully connected layer to get latent vector
        return x

# Define the Decoder class, which reconstructs an image from a latent vector.
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):  # Constructor where latent_dim matches the encoder's output
        super(Decoder, self).__init__()
        # Fully connected layer to expand the latent vector back to a 256 x 16 x 16 feature map
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        # Define 3 deconvolutional (transpose convolution) layers to upsample the feature map and reconstruct the image
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Upsampling to 128 channels
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsampling to 64 channels
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # Final layer outputs 3 channels (RGB image)

    def forward(self, x):
        x = self.fc(x)  # Expand the latent vector
        x = x.view(x.size(0), 256, 16, 16)  # Reshape to match the feature map shape
        # Apply ReLU activation after each deconvolution and sigmoid at the end for pixel intensity normalization
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Sigmoid ensures pixel values are between 0 and 1
        return x

# Define the SegmentationDecoder class, which outputs a segmentation mask from a latent vector.
class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3):  # Constructor, num_classes defines the number of segmentation classes (e.g., road, sidewalk, etc.)
        super(SegmentationDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)  # Expand the latent vector back to a feature map
        # Define deconvolution layers similar to the decoder, but with the number of output channels equal to num_classes
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)  # Expand the latent vector
        x = x.view(x.size(0), 256, 16, 16)  # Reshape to feature map size
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        # Apply softmax to output probabilities across segmentation classes
        x = torch.softmax(self.deconv3(x), dim=1)
        return x

# Define a dataset class for loading snowy images and optionally their corresponding segmentation masks.
class SnowyDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):  # Constructor, image_dir is the directory of images, mask_dir is optional for segmentation masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)  # List all images in the directory
        self.transform = transform  # Optional image transformation (resize, normalize, etc.)
        if mask_dir:
            self.mask_filenames = os.listdir(mask_dir)  # List all masks if available
            # Ensure the number of images matches the number of masks, truncate masks if necessary
            if len(self.image_filenames) != len(self.mask_filenames):
                print(f"Warning: {len(self.image_filenames)} images and {len(self.mask_filenames)} masks found.")
                self.mask_filenames = self.mask_filenames[:len(self.image_filenames)]

    def __len__(self):
        return len(self.image_filenames)  # Return the number of images in the dataset

    def __getitem__(self, idx):
        # Load image and apply transformation if specified
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # If masks are provided, load the corresponding mask
        if self.mask_dir:
            if idx < len(self.mask_filenames):  # Ensure mask index is valid
                mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
                mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
                if self.transform:
                    mask = self.transform(mask)
                return image, mask  # Return image and mask
            else:
                print(f"Mask for image {self.image_filenames[idx]} not found.")
                return image, torch.zeros_like(image)  # Return a placeholder if no mask is found
        else:
            return image  # Return image only if no mask is provided

# Define image transformations (resize and normalize for input to the model)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Dataset and DataLoader for snowy images (used for autoencoder training)
snowy_dataset = SnowyDataset(image_dir='cam6/small_snow', transform=transform)
snowy_dataloader = DataLoader(snowy_dataset, batch_size=32, shuffle=True, num_workers=4)

# Dataset and DataLoader for snowy images with masks (used for segmentation training)
# NOTE: "masks" here are actually the clear (non-snowy) versions of the same
# scenes, loaded as grayscale — a placeholder stand-in for real per-pixel
# class-label masks, not true segmentation ground truth.
mask_dataset = SnowyDataset(image_dir='cam6/small_snow', mask_dir='cam6/small_clear', transform=transform)
clear_mask_dataloader = DataLoader(mask_dataset, batch_size=32, shuffle=True, num_workers=4)

# Autoencoder training (encoder + decoder)
encoder = Encoder()  # Initialize the encoder
decoder = Decoder()  # Initialize the decoder
autoencoder = nn.Sequential(encoder, decoder)  # Combine encoder and decoder into an autoencoder
criterion = nn.MSELoss()  # Mean Squared Error (MSE) loss function for reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)  # Adam optimizer for training the autoencoder

# Training loop for the autoencoder
for epoch in range(20):  # Train for 20 epochs
    for snowy_images in snowy_dataloader:
        optimizer.zero_grad()  # Zero out the gradients from the previous step
        reconstructed = autoencoder(snowy_images)  # Pass images through the autoencoder
        loss = criterion(reconstructed, snowy_images)  # Compute reconstruction loss (how different is
        # the reconstructed image from the original)
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights using the optimizer

    # Print the loss for each epoch to monitor training progress
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Step 2: Train the Segmentation Decoder (uses frozen encoder weights)
# `eval()` only disables dropout/batchnorm training behavior — the encoder's
# weights are actually kept frozen because `optimizer_segmentation` below is
# only given `segmentation_decoder.parameters()`, so no gradient step ever
# touches the encoder even though gradients still flow through it.
encoder.eval()
segmentation_decoder = SegmentationDecoder()  # Initialize the segmentation decoder
criterion_segmentation = nn.CrossEntropyLoss()  # Cross-Entropy loss for multi-class segmentation
optimizer_segmentation = optim.Adam(segmentation_decoder.parameters(), lr=1e-4)  # Adam optimizer for segmentation training

# Function to train the segmentation decoder with frozen encoder
def train_segmentation(encoder, segmentation_decoder, dataloader, criterion, optimizer):
    for epoch in range(20):  # Train for 20 epochs
        for snowy_images, clear_masks in dataloader:
            optimizer.zero_grad()  # Zero out the gradients

            # Pass images through the encoder to obtain latent representations
            latent_vectors = encoder(snowy_images)
            # Pass latent vectors through segmentation decoder to get segmentation masks
            segmentation_output = segmentation_decoder(latent_vectors)
            # Compute segmentation loss by comparing predicted masks to ground truth masks
            loss = criterion(segmentation_output, clear_masks)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights in the segmentation decoder

        # Print segmentation loss for each epoch to monitor progress
        print(f'Epoch {epoch + 1}, Segmentation Loss: {loss.item()}')

# Train the segmentation decoder using the frozen encoder
train_segmentation(encoder, segmentation_decoder, clear_mask_dataloader, criterion_segmentation, optimizer_segmentation)