from torchmetrics.functional import (mean_absolute_error as mae_metric, mean_squared_error as mse_metric, 
                                     structural_similarity_index_measure as ssim_metric)
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_metric
import numpy as np
def compute_mse_per_checkpoint(denoised,images):
    
        # Iter over every image
        for i in range(len(denoised)):
                # The original image is expanded (literally repeated) n_points times
                original_expanded = images[i].unsqueeze(0).expand_as(denoised[i]).contiguous()
                
                mse_per_image = []
                # Iter over every checkpoint of the image
                for j in range(denoised[i].shape[0]):
                        checkpoint = denoised[i][j:j+1]  # The j-th checkpoint is extracted (keeping batch dimension)
                        orig = original_expanded[j:j+1]

                        if checkpoint.min() < 0:
                            checkpoint = checkpoint*0.5 + 0.5
                            print('checkpoint min: ', checkpoint.min()) #Debug
                        if orig.min() < 0:
                            orig = orig*0.5 + 0.5
                            print('orig min: ', orig.min())  #Debug

                        mse = mse_metric(checkpoint, orig)
                        mse_per_image.append(np.mean(mse.item()))
        
        return np.array(mse_per_image)
    

def compute_mae_per_checkpoint(denoised,images):

    # Iter over every image
    for i in range(len(denoised)):
            # The original image is expanded (literally repeated) n_points times
            original_expanded = images[i].unsqueeze(0).expand_as(denoised[i]).contiguous()
            
            mae_per_image = []
            # Iter over every checkpoint of the image
            for j in range(denoised[i].shape[0]):
                    checkpoint = denoised[i][j:j+1]  # The j-th checkpoint is extracted (keeping batch dimension)
                    orig = original_expanded[j:j+1]

                    if checkpoint.min() < 0:
                        checkpoint = checkpoint*0.5 + 0.5
                        print('checkpoint min: ', checkpoint.min()) #Debug
                    if orig.min() < 0:
                        orig = orig*0.5 + 0.5
                        print('orig min: ', orig.min())  #Debug

                    mae = mae_metric(checkpoint, orig)
                    mae_per_image.append(np.mean(mae.item()))
    
    return np.array(mae_per_image)

def compute_ssim_per_checkpoint(denoised,images):

    # Iter over every image
    for i in range(len(denoised)):
            # The original image is expanded (literally repeated) n_points times
            original_expanded = images[i].unsqueeze(0).expand_as(denoised[i]).contiguous()
            
            ssim_per_image = []
            # Iter over every checkpoint of the image
            for j in range(denoised[i].shape[0]):
                    checkpoint = denoised[i][j:j+1]  # The j-th checkpoint is extracted (keeping batch dimension)
                    orig = original_expanded[j:j+1]

                    if checkpoint.min() < 0:
                        checkpoint = checkpoint*0.5 + 0.5
                        print('checkpoint min: ', checkpoint.min()) #Debug
                    if orig.min() < 0:
                        orig = orig*0.5 + 0.5
                        print('orig min: ', orig.min())  #Debug

                    ssim = ssim_metric(checkpoint, orig, data_range = 1.0)
                    ssim_per_image.append(np.mean(ssim.item()))
    return np.array(ssim_per_image)

def compute_psnr_per_checkpoint(denoised,images):

    # Iter over every image
    for i in range(len(denoised)):
            # The original image is expanded (literally repeated) n_points times
            original_expanded = images[i].unsqueeze(0).expand_as(denoised[i]).contiguous()
            
            psnr_per_image = []
            # Iter over every checkpoint of the image
            for j in range(denoised[i].shape[0]):
                    checkpoint = denoised[i][j:j+1]  # The j-th checkpoint is extracted (keeping batch dimension)
                    orig = original_expanded[j:j+1]

                    if checkpoint.min() < 0:
                        checkpoint = checkpoint*0.5 + 0.5
                        print('checkpoint min: ', checkpoint.min()) #Debug
                    if orig.min() < 0:
                        orig = orig*0.5 + 0.5
                        print('orig min: ', orig.min())  #Debug

                    psnr = psnr_metric(checkpoint, orig, data_range = 1.0)
                    psnr_per_image.append(np.mean(psnr.item()))
    
    return np.array(psnr_per_image)

def compute_psnr(denoised_image, image):

    if denoised_image.min() < 0:
        denoised_image = denoised_image*0.5 + 0.5

    if image.min() < 0:
        image = image*0.5 + 0.5

    psnr = psnr_metric(denoised_image, image, data_range = 1.0)
    return psnr.item()

def compute_ssim(denoised_image, image):

    if denoised_image.min() < 0:
        denoised_image = denoised_image*0.5 + 0.5

    if image.min() < 0:
        image = image*0.5 + 0.5
        
    ssim = ssim_metric(denoised_image, image, data_range = 1.0)
    return ssim.item()