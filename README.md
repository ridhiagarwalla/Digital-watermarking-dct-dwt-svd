Here's a step-by-step explanation of the entire process:

---

### *Step 1: Embedding the Watermark*

1. *Command Execution:*  
   You run the command:  
   bash
   python main.py --origin cover.jpg --ouput watermarked.jpg
     
   *(Note: Make sure you use the correct flag --output if that's how your script is set up.)*

2. *Algorithm Selection:*  
   - *Prompt:* The program asks you to choose a type: “DCT” or “DWT” or "SVD".  
   - *Action:* You select one based on your preference (e.g., DCT).

3. *Operation Choice:*  
   - *Prompt:* Next, you are asked to choose an option: “embedding” or “extracting”.  
   - *Action:* You choose “embedding”.

4. *Watermark Embedding Process:*  
   - *What Happens Internally:*  
     - The chosen algorithm (DCT or DWT or SVD) takes the cover image and the watermark image.
     - It applies a transformation (like the Discrete Cosine Transform or Discrete Wavelet Transform) to break down the cover image.
     - It then embeds the watermark (or signature) into certain frequency components by slightly modifying them.
     - Finally, it reconstructs the image with the watermark embedded.
   - *Outcome:* A watermarked image is generated and saved as watermarked.jpg.

---

### *Step 2: Attacking the Watermarked Image*

1. *Command Execution:*  
   You run the command:  
   bash
   python main.py --origin watermarked.jpg --ouput attacked.jpg
     
   *(Again, check your flag spelling; it should be --output.)*

2. *Attack Option:*  
   - *Prompt:* The program now asks you to choose “Attack” as the type.
   - *Action:* You select “Attack”.

3. *Choosing an Attack:*  
   - *Prompt:* The script displays a list of attack options such as blur, rotate, noise addition, cropping, etc.
   - *Action:* You select one attack (e.g., “blur”).

4. *Attack Process:*  
   - *What Happens Internally:*  
     - The chosen attack algorithm modifies the watermarked image (e.g., applies a Gaussian blur).
     - This simulates real-world scenarios where the image might be tampered with, compressed, or distorted.
   - *Outcome:* An attacked version of the watermarked image is created and saved as attacked.jpg.

---

### *Step 3: Extracting the Watermark from the (Attacked) Image*

1. *Command Execution:*  
   You run the command:  
   bash
   python main.py --origin attacked.jpg --ouput signature.jpg
     
   *(Replace attacked.jpg with the image from which you want to extract the watermark.)*

2. *Algorithm Selection:*  
   - *Prompt:* The program asks you to choose a type (DCT or DWT or SVD).  
   - *Action:* You select the same algorithm you used during embedding (this is important because the extraction process relies on the same transform).

3. *Operation Choice:*  
   - *Prompt:* Next, you choose “extracting” from the given options.
   - *Action:* You select “extracting”.

4. *Watermark Extraction Process:*  
   - *What Happens Internally:*  
     - The algorithm applies the inverse of the transform (DCT or DWT) on the attacked image.
     - It tries to recover the modified components where the watermark was embedded.
     - It extracts the watermark (or signature) from these components.
   - *Outcome:* An extracted watermark image (signature) is generated and saved as signature.jpg.

---

### *Step 4: Checking the Extracted Watermark*

1. *Visual Inspection:*  
   - Open signature.jpg to see if the watermark is visible.
   - Compare it to the original watermark image to check for similarity.

2. *Robustness Analysis (Optional):*  
   - Often, the extraction process involves measuring the correlation between the original watermark and the extracted one.
   - A high correlation value indicates that the watermark is robust even after the attack.
   - If the attack is too severe, the extracted watermark might be distorted or partially lost.

---

### *Summary of the Process:*

1. *Embed:*  
   - You embed a watermark into a cover image.
   - The watermarked image is created.

2. *Attack:*  
   - The watermarked image is intentionally degraded (attacked) to simulate real-world distortions.
   - The attacked image is created.

3. *Extract:*  
   - The watermark is then extracted from the attacked image.
   - The extraction process helps you verify whether the watermark remains detectable and robust after the attack.

By following these steps, you can evaluate the effectiveness of your watermarking algorithm and determine its resilience against various image manipulations.

Let me know if you need any further details or clarifications!
