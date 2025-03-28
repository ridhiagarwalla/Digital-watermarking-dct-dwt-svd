import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from attack import Attack
from watermark import Watermark


class DCT_Watermark(Watermark):
    def __init__(self):
        self.Q = 10
        self.size = 2
        self.sig_size = 100  # Added signature size for consistency

    def inner_embed(self, B: np.ndarray, signature):
        sig_size = self.sig_size
        size = self.size
        w, h = B.shape[:2]
        embed_pos = [(0, 0)]

        if w > 2 * sig_size * size:
            embed_pos.append((w - sig_size * size, 0))
        if h > 2 * sig_size * size:
            embed_pos.append((0, h - sig_size * size))
        if len(embed_pos) == 3:
            embed_pos.append((w - sig_size * size, h - sig_size * size))

        for x, y in embed_pos:
            for i in range(x, x + sig_size * size, size):
                for j in range(y, y + sig_size * size, size):
                    v = np.float32(B[i:i + size, j:j + size])
                    v = cv2.dct(v)
                    v[size - 1, size - 1] = self.Q * signature[((i - x) // size) * sig_size + (j - y) // size]
                    v = cv2.idct(v)

                    # Clamp pixel values between [0, 255]
                    v = np.clip(v, 0, 255)
                    B[i:i + size, j:j + size] = v

        return B

    def inner_extract(self, B):
        sig_size = self.sig_size
        size = self.size
        ext_sig = np.zeros(sig_size ** 2, dtype=int)

        for i in range(0, sig_size * size, size):
            for j in range(0, sig_size * size, size):
                v = cv2.dct(np.float32(B[i:i + size, j:j + size]))
                if v[size - 1, size - 1] > self.Q / 2:
                    ext_sig[(i // size) * sig_size + (j // size)] = 1
        return [ext_sig]


if __name__ == "__main__":
    # Load cover and watermark images
    img = cv2.imread("images/cover.jpg", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("images/watermark.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Error: Could not load cover image.")
    if wm is None:
        raise FileNotFoundError("Error: Could not load watermark image.")

    # Initialize DCT Watermark object
    dct = DCT_Watermark()

    # Embed watermark
    signature = (wm > 128).flatten().astype(int)
    watermarked_img = dct.inner_embed(img.copy(), signature)
    cv2.imwrite("images/watermarked.jpg", watermarked_img)

    # Simulate an attack (convert to grayscale as example)
    attacked_img = Attack.gray(watermarked_img)
    cv2.imwrite("images/attacked_watermarked.jpg", attacked_img)

    # Extract watermark
    extracted_signature = dct.inner_extract(attacked_img)[0].reshape((dct.sig_size, dct.sig_size)) * 255
    extracted_signature = extracted_signature.astype(np.uint8)
    cv2.imwrite("images/extracted_signature.jpg", extracted_signature)

    # Resize extracted watermark if dimensions differ
    if wm.shape != extracted_signature.shape:
        extracted_signature = cv2.resize(
            extracted_signature, (wm.shape[1], wm.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    # Convert both images to float64 for consistency
    wm_float = wm.astype(np.float64)
    extracted_signature_float = extracted_signature.astype(np.float64)

    # Calculate PSNR and SSIM with specified data range (since pixel values range from 0 to 255)
    psnr_value = psnr(wm_float, extracted_signature_float, data_range=255)
    ssim_value = ssim(wm_float, extracted_signature_float, data_range=255)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

