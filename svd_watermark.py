import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from attack import Attack  # Ensure Attack is defined in attack.py
from watermark import Watermark

class SVD_Watermark(Watermark):
    def __init__(self):
        super().__init__()
        self.alpha = 0.05  # Strength factor for watermark embedding

    def embed(self, cover_img: np.ndarray, watermark_img: np.ndarray):
        """
        Embeds the watermark into the cover image using Singular Value Decomposition (SVD).
        """
        if cover_img is None:
            raise FileNotFoundError("Error: Cover image not found or could not be loaded.")
        if watermark_img is None:
            raise ValueError("Error: Watermark image is None in SVD embedding.")
        # Ensure cover image is grayscale
        if len(cover_img.shape) == 3:
            cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
        sig_size = min(cover_img.shape[0], cover_img.shape[1])  # Set signature size dynamically
        watermark_resized = cv2.resize(watermark_img, (sig_size, sig_size))
        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(cover_img.astype(np.float32), full_matrices=False)
        # Embed watermark into singular values
        S[:sig_size] += self.alpha * watermark_resized.flatten()[:sig_size]
        # Reconstruct the watermarked image
        watermarked_img = np.dot(U, np.dot(np.diag(S), Vt))
        watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
        return watermarked_img

    def extract(self, watermarked_img: np.ndarray):
        """
        Extracts the watermark from the watermarked image using SVD.
        """
        if watermarked_img is None:
            raise FileNotFoundError("Error: Watermarked image not found or could not be loaded.")

        # Ensure the image is grayscale
        if len(watermarked_img.shape) == 3:
            watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
        sig_size = min(watermarked_img.shape[0], watermarked_img.shape[1])  # Set signature size
        # Perform SVD on the watermarked image
        U, S, Vt = np.linalg.svd(watermarked_img.astype(np.float32), full_matrices=False)
        # Extract watermark
        extracted_signature = (S[:sig_size] / self.alpha)
        # Ensure the number of elements is a perfect square before reshaping
        sqrt_size = int(np.sqrt(len(extracted_signature)))
        extracted_signature = extracted_signature[:sqrt_size**2]
        extracted_signature = extracted_signature.reshape(sqrt_size, sqrt_size).astype(np.uint8)
        return extracted_signature


if __name__ == "__main__":
    try:
        # Load cover image
        img = cv2.imread("images/cover.jpg")
        if img is None:
            raise FileNotFoundError("Error: Cover image not found or could not be loaded.")
        # Load watermark image in grayscale
        wm = cv2.imread("images/watermark.jpg", cv2.IMREAD_GRAYSCALE)
        if wm is None:
            raise FileNotFoundError("Error: Watermark image not found or could not be loaded.")

        # Initialize SVD Watermarking
        svd = SVD_Watermark()

        # Embed watermark
        watermarked_img = svd.embed(img, wm)
        cv2.imwrite("images/watermarked.jpg", watermarked_img)

        # Apply attack (convert to grayscale or any other attack method)
        img_attacked = cv2.imread("images/watermarked.jpg")
        if img_attacked is None:
            raise FileNotFoundError("Error: Watermarked image for attack not found.")

        # Apply the attack method (ensure Attack.gray is defined in attack.py)
        img_attacked = Attack.gray(img_attacked)  # Attack should convert to grayscale
        cv2.imwrite("images/watermarked_attacked.jpg", img_attacked)

        # Extract watermark from attacked image
        img_extracted = cv2.imread("images/watermarked_attacked.jpg", cv2.IMREAD_GRAYSCALE)
        if img_extracted is None:
            raise FileNotFoundError("Error: Attacked watermarked image not found or could not be loaded.")

        extracted_signature = svd.extract(img_extracted)
        cv2.imwrite("images/extracted_signature.jpg", extracted_signature)

        # Calculate PSNR and SSIM between original watermark and extracted watermark
        psnr_value = psnr(wm, extracted_signature)
        ssim_value = ssim(wm, extracted_signature)

        print(f"PSNR: {psnr_value:.2f}")
        print(f"SSIM: {ssim_value:.4f}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")