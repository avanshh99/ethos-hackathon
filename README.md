# Round 1 solution for ML statement of Ethos Hackathon

## Approach:

1. Frame Extraction: Extract frames from the video using OpenCV to isolate images at regular intervals for further processing.

2. Face Detection: Detect faces in each extracted frame using MTCNN, cropping the face areas for enhancement.

3. Preprocessing: Align and normalize the detected faces to ensure consistency in facial orientation and size before sending them to enhancement models.

4. Facial Enhancement (GAN-based Models): Pass the cropped faces through models like CodeFormer and GFPGAN to restore and enhance facial details, improving resolution and clarity.

5. Feature Reconstruction: Enhance facial features using deep learning techniques embedded in the models, reducing blur and improving overall image quality.

6. Post-Processing: Combine the reconstructed faces with the original frames, ensuring the faces are enhanced while preserving frame continuity for any further analysis.

## Results

#### The result folder contains the comparative results tried on the video [content/market.mp4](https://github.com/Ha4sh-447/ethos-hackathon/blob/main/content/market.mp4).

The result folder contains codeFormer_result and gfpgan_result respectively.
[result/gfpgan_result/cmp](result/gfpgan_result/cmp) and [result/coderformer_result/compare](result/codeformer_result/compare) contains side by side comparison of extracted and enhanced image

### Extracted Frame

<img src="https://cdn.discordapp.com/attachments/1284411927115595786/1289821731502161962/frame_0_1.jpg?ex=66fa3798&is=66f8e618&hm=98147268bab760d8c16d128d9785b3bf6224fe15ac94aedcf97c17b4a411a8ee&" alt="Extracted Frame" width="400"/>

### Here are some of the results from the model

#### Image 1

<img src="https://iili.io/dZtKBzF.png" alt="Comparative Result" width="400"/>

#### Image 2

<img src="https://iili.io/dZ85H5F.jpg" alt="Comparative Result 2" width="400"/>

#### Image 3

<img src="https://iili.io/dZ8U92V.png" alt="Comparative Result 3" width="400"/>
