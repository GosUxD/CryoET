import torch
import numpy as np
import torch.fft
#from utils import load_tomogram

def bandpass_filter(image, low_cutoff=None, high_cutoff=None):
    """
    Applies a bandpass filter to a 2D or 3D image in Fourier space.

    Args:
        image (torch.Tensor): Input 2D or 3D image tensor.
                              Shape: (H, W) for 2D, (D, H, W) for 3D.
        low_cutoff (float, optional): Lower cutoff frequency (normalized, 0 to 1).
                                      Randomized if None.
        high_cutoff (float, optional): Higher cutoff frequency (normalized, 0 to 1).
                                       Randomized if None.

    Returns:
        torch.Tensor: Bandpass-filtered image.
    """
    # Randomize cutoff frequencies if not provided
    if low_cutoff is None:
        low_cutoff = np.random.uniform(0.01, 0.15)
    if high_cutoff is None:
        high_cutoff = np.random.uniform(0.7, 0.8)

    # Compute FFT and shift zero frequency to the center
    fft_image = torch.fft.fftn(image)
    fft_shifted = torch.fft.fftshift(fft_image)

    # Generate frequency grid
    if image.ndim == 2:
        H, W = image.shape
        y, x = torch.meshgrid(torch.arange(-H // 2, H // 2), torch.arange(-W // 2, W // 2), indexing="ij")
        radius = torch.sqrt((x / (W / 2))**2 + (y / (H / 2))**2)
    elif image.ndim == 3:
        D, H, W = image.shape
        z, y, x = torch.meshgrid(
            torch.arange(-D // 2, D // 2),
            torch.arange(-H // 2, H // 2),
            torch.arange(-W // 2, W // 2),
            indexing="ij"
        )
        radius = torch.sqrt((x / (W / 2))**2 + (y / (H / 2))**2 + (z / (D / 2))**2)
    else:
        raise ValueError("Input image must be 2D or 3D.")

    # Generate bandpass mask
    mask = (radius >= low_cutoff) & (radius <= high_cutoff)

    # Apply mask in Fourier space
    filtered_fft = fft_shifted * mask

    # Inverse FFT to return to spatial domain
    filtered_image = torch.fft.ifftshift(filtered_fft)
    filtered_image = torch.fft.ifftn(filtered_image).real

    return filtered_image

def frequency_noise(image):
    # Compute FFT
    fft_image = torch.fft.fftn(image)
    fft_shifted = torch.fft.fftshift(fft_image)

    std_dev = np.random.uniform(0.07, 0.12)
    weight = np.random.uniform(0.001, 0.005)

    # Add Gaussian noise
    noise = torch.randn_like(fft_shifted.real) * std_dev + 1j * torch.randn_like(fft_shifted.imag) * std_dev
    noisy_fft = fft_shifted + noise * weight

    # Inverse FFT
    noisy_image = torch.fft.ifftshift(noisy_fft)
    noisy_image = torch.fft.ifftn(noisy_image).real

    return noisy_image

def missing_wedge(image, wedge_angle=None):
    """
    Applies a missing wedge mask to simulate anisotropic frequency loss in Fourier space.

    Args:
        image (torch.Tensor): Input image tensor of shape (H, W) (for 2D) or (D, H, W) (for 3D).
        wedge_angle (float, optional): Angle of the missing wedge (in degrees). 
                                       If None, a random angle between 15° and 45° is used.

    Returns:
        torch.Tensor: Image with missing wedge effect.
    """
    if wedge_angle is None:
        wedge_angle = np.random.uniform(15, 45)
    
    is_3d = (image.ndim == 3)  # Check if the input is 3D

    # Compute FFT
    fft_image = torch.fft.fftn(image)
    fft_shifted = torch.fft.fftshift(fft_image)

    # Generate frequency grid and wedge mask
    if is_3d:
        D, H, W = image.shape
        z, y, x = torch.meshgrid(
            torch.arange(-D // 2, D // 2),
            torch.arange(-H // 2, H // 2),
            torch.arange(-W // 2, W // 2),
            indexing="ij"
        )
        theta = torch.atan2(torch.sqrt(y**2 + x**2).float(), z.float())  # Angle from z-axis
    else:
        H, W = image.shape
        y, x = torch.meshgrid(torch.arange(-H // 2, H // 2), torch.arange(-W // 2, W // 2))
        theta = torch.atan2(y.float(), x.float())  # Angle in 2D plane

    # Wedge mask: frequencies outside the wedge angle are zeroed
    wedge_mask = (torch.abs(theta) > torch.deg2rad(torch.tensor(wedge_angle)))

    # Apply the wedge mask
    fft_masked = fft_shifted * wedge_mask

    # Inverse FFT
    masked_image = torch.fft.ifftshift(fft_masked)
    masked_image = torch.fft.ifftn(masked_image).real

    return masked_image

def band_dropout(image, dropout_fraction=0.1, band='high'):
    if not (0 <= dropout_fraction <= 1):
        raise ValueError("dropout_fraction must be between 0 and 1.")

    if np.random.rand() > 0.5:
        band = 'high'
        dropout_fraction = np.random.uniform(0.6, 0.7)
    else:
        band = 'low'
        dropout_fraction = np.random.uniform(0.01, 0.2)
        
    # Compute FFT and shift zero frequency to the center
    fft_image = torch.fft.fftn(image)
    fft_shifted = torch.fft.fftshift(fft_image)

    # Generate frequency grid
    if image.ndim == 2:
        rows, cols = fft_shifted.shape
        center_row, center_col = rows // 2, cols // 2
        Y, X = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        distance = torch.sqrt((X - center_col)**2 + (Y - center_row)**2)
        max_distance = torch.max(distance)
    elif image.ndim == 3:
        depth, rows, cols = fft_shifted.shape
        center_depth, center_row, center_col = depth // 2, rows // 2, cols // 2
        Z, Y, X = torch.meshgrid(
            torch.arange(depth), torch.arange(rows), torch.arange(cols), indexing="ij"
        )
        distance = torch.sqrt((X - center_col)**2 + (Y - center_row)**2 + (Z - center_depth)**2)
        max_distance = torch.max(distance)
    else:
        raise ValueError("Input image must be 2D or 3D.")

    # Create the dropout mask
    if band == 'high':
        cutoff = (1 - dropout_fraction) * max_distance
        mask = distance < cutoff  # Retain low frequencies
    elif band == 'low':
        cutoff = dropout_fraction * max_distance
        mask = distance > cutoff  # Retain high frequencies
    else:
        raise ValueError("Band must be either 'high' or 'low'.")

    # Apply the mask
    fft_masked = fft_shifted * mask

    # Inverse FFT to return to spatial domain
    masked_image = torch.fft.ifftshift(fft_masked)
    masked_image = torch.fft.ifftn(masked_image).real

    return masked_image

def randomize_phase(image):
    # Compute FFT
    fft_image = torch.fft.fftn(image)

    # Extract magnitude and phase
    magnitude = torch.abs(fft_image)
    phase = torch.angle(fft_image)

    # Randomize phase
    random_phase = torch.rand_like(phase) * 2 * np.pi - np.pi
    phase = 0.95 * phase + 0.05 * random_phase
    randomized_fft = magnitude * torch.exp(1j * phase)

    # Inverse FFT
    randomized_image = torch.fft.ifftn(randomized_fft).real

    return randomized_image



# import matplotlib.pyplot as plt
# from utils import load_coordinates_xyz
# import sys

# coordinates = load_coordinates_xyz("TS_5_4", "thyroglobulin")
# example = 1
# x, y, z = int(coordinates[example]['x'] // 10), int(coordinates[example]['y'] // 10), int(coordinates[example]['z'] // 10)
# dlta = 36
# shift = 50

# x += shift

# tomogram = load_tomogram("TS_5_4", type='denoised')
# image = torch.tensor(tomogram[z, y-dlta:y+dlta, x-dlta:x+dlta]).float()

# # Apply augmentations
# bandpass_filtered = bandpass_filter(image)
# noisy_image = frequency_noise(image)
# wedge_image = missing_wedge(image)
# band_dropout = band_dropout(image)
# randomized_phase = randomize_phase(image)

# fft_image = torch.fft.fft2(image)
# fft_image = torch.fft.fftshift(fft_image).real[dlta-25:dlta+25, dlta-25:dlta+25]

# # Plot results
# plt.figure(figsize=(18, 9))
# plt.subplot(331), plt.imshow(image.numpy(), cmap='gray'), plt.title('Original')
# plt.subplot(332), plt.imshow(bandpass_filtered.numpy(), cmap='gray'), plt.title('Bandpass Filtered')
# plt.subplot(333), plt.imshow(noisy_image.numpy(), cmap='gray'), plt.title('Noise Injected')
# plt.subplot(334), plt.imshow(wedge_image, cmap='gray'), plt.title('Missing Wedge')
# plt.subplot(335), plt.imshow(band_dropout.numpy(), cmap='gray'), plt.title('Band Dropout')
# plt.subplot(336), plt.imshow(randomized_phase.numpy(), cmap='gray'), plt.title('Random Phase')
# plt.subplot(337), plt.imshow(np.log(1 + np.abs(fft_image.numpy())), cmap='gray'), plt.title('Original FFT')
# plt.show()



