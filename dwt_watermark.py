import math
import cv2
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class DWT_Watermark:
    def __init__(self):
        self.sig_size = 32  # Signature size (32x32)

    def __gene_embed_space(self, vec):
        shape = vec.shape
        vec = vec.flatten()
        combo_neg_idx = np.array([1 if val < 0 else 0 for val in vec])
        vec_pos = np.abs(vec)
        int_part = np.floor(vec_pos)
        frac_part = np.round(vec_pos - int_part, 2)
        bi_int_part = []
        for val in int_part:
            bi = list(bin(int(val))[2:])  # Convert to binary string without '0b'
            padded_bi = [0] * (16 - len(bi)) + list(map(int, bi))  # Pad to 16 bits
            bi_int_part.append(np.array(padded_bi, dtype=np.uint16))

        bi_int_part = np.array(bi_int_part)
        sig = bi_int_part[:, 10].reshape(shape)  # Extract the 11th bit for embedding

        return bi_int_part, frac_part.reshape(shape), combo_neg_idx.reshape(shape), sig

    def __embed_sig(self, bi_int_part, frac_part, combo_neg_idx, signature):
        shape = frac_part.shape
        frac_part = frac_part.flatten()
        combo_neg_idx = combo_neg_idx.flatten()
        signature = np.resize(signature, bi_int_part.shape[0])  # Resize signature if needed
        # Embed signature into the 11th bit
        for i in range(len(bi_int_part)):
            bi_int_part[i][10] = signature[i]
        em_int_part = np.array([int(''.join(map(str, bi)), 2) for bi in bi_int_part])
        em_combo = em_int_part + frac_part
        em_combo = np.where(combo_neg_idx == 1, -em_combo, em_combo)
        return em_combo.reshape(shape)

    def __extract_sig(self, HH_3, siglen):
        _, _, _, ori_sig = self.__gene_embed_space(HH_3)
        ext_sig = ori_sig.flatten()[:siglen]  # Extract required signature length
        return ext_sig

    def inner_embed(self, B, signature):
        # Convert to grayscale if image has 3 channels (RGB)
        if B.ndim == 3:
            B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
        w, h = B.shape  # Now B will be 2D (w, h)
        cropped_B = B[:32 * (w // 32), :32 * (h // 32)]
        # Perform 4-level DWT decomposition
        coeffs = pywt.wavedec2(cropped_B, 'haar', level=4)
        LL_4, (HL_4, LH_4, HH_4) = coeffs[0], coeffs[1]
        bi_int_part, frac_part, combo_neg_idx, _ = self.__gene_embed_space(HH_4)
        HH_4_embedded = self.__embed_sig(bi_int_part, frac_part, combo_neg_idx, signature)
        # Reconstruct using inverse DWT
        coeffs[1] = (HL_4, LH_4, HH_4_embedded)
        reconstructed = pywt.waverec2(coeffs, 'haar')
        B[:reconstructed.shape[0], :reconstructed.shape[1]] = reconstructed
        return B

    def inner_extract(self, B):
        # Convert to grayscale if image has 3 channels (RGB)
        if B.ndim == 3:
            B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
        w, h = B.shape  # Ensure B is now 2D (w, h)
        cropped_B = B[:32 * (w // 32), :32 * (h // 32)]
        # Perform 4-level DWT decomposition for extraction
        coeffs = pywt.wavedec2(cropped_B, 'haar', level=4)
        _, (HL_4, LH_4, HH_4) = coeffs[0], coeffs[1]
        extracted_sig = self.__extract_sig(HH_4, self.sig_size ** 2)
        return extracted_sig.reshape((self.sig_size, self.sig_size))

    def embed(self, img, wm):
        signature = (wm > 128).astype(int).flatten()
        watermarked_img = self.inner_embed(img.copy(), signature)
        return watermarked_img

    def extract(self, img):
        extracted_signature = self.inner_extract(img)
        return (extracted_signature * 255).astype(np.uint8)


if __name__ == "__main__":
    # Load images
    img = cv2.imread("images/cover.jpg", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("images/watermark.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None or wm is None:
        print("Error: Images not found. Check the paths.")
        exit()

    # Resize watermark to match signature size
    wm = cv2.resize(wm, (32, 32))
    dwt = DWT_Watermark()

    # Embed watermark
    watermarked_img = dwt.embed(img, wm)
    cv2.imwrite("images/watermarked.jpg", watermarked_img)

    # Simulate an attack (optional)
    attacked_img = cv2.imread("images/watermarked.jpg", cv2.IMREAD_GRAYSCALE)

    # Extract watermark
    extracted_wm = dwt.extract(attacked_img)
    cv2.imwrite("images/signature.jpg", extracted_wm)

    # Calculate PSNR and SSIM
    wm_float = wm.astype(np.float64)
    extracted_wm_float = extracted_wm.astype(np.float64)
    psnr_value = psnr(wm_float, extracted_wm_float, data_range=255)
    ssim_value = ssim(wm_float, extracted_wm_float, data_range=255)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")