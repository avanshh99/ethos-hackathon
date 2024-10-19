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

## Added Features

Real-Time Monitoring:
Implemented real-time monitoring using OpenCV2 and the face_recognition library. This allows the system to detect and recognize faces in real-time from live video feeds or pre-recorded videos.

Vector Database for Face Recognition:
Integrated a vector database for face recognition to detect and identify suspects from a pre-existing database. If the person is known, the system flags the match and provides relevant information.

3D Face Reconstruction (3DDFA-V3):
Added 3D face reconstruction using 3DDFA-V3, where the system can construct a 3D mesh of the detected face. By using RetinaFace for enhanced image processing, users can view the 3D model of the face in a 360-degree view.

UI/UX (Facial-Police):
Developed a user-friendly UI/UX platform called "Facial-Police" that facilitates the reconstruction of faces from CCTV footage or other video scenarios. The interface allows seamless access, making it easy for users to upload footage, reconstruct faces, and analyze results without hassle.


### Extracted Frame

<img src="https://cdn.discordapp.com/attachments/1284411927115595786/1289821731502161962/frame_0_1.jpg?ex=66fa3798&is=66f8e618&hm=98147268bab760d8c16d128d9785b3bf6224fe15ac94aedcf97c17b4a411a8ee&" alt="Extracted Frame" width="400"/>

### Here are some of the results from the model

#### Image 1

<img src="https://iili.io/dZtKBzF.png" alt="Comparative Result" width="400"/>

#### Image 2

<img src="https://iili.io/dZ85H5F.jpg" alt="Comparative Result 2" width="400"/>

#### Image 3

<img src="https://iili.io/dZ8U92V.png" alt="Comparative Result 3" width="400"/>

### UI 
<img src="https://i.postimg.cc/t4b4G1WF/Whats-App-Image-2024-10-19-at-20-56-21-8ed59cc4.jpg" alt="Comparative Result 3" width="400"/>
<img src="https://i.postimg.cc/8P5qJTGx/Whats-App-Image-2024-10-19-at-20-56-20-065e44c92222222222222.jpg" alt="Comparative Result 3" width="400"/>
<img src="blob:https://web.whatsapp.com/a057bf74-f3a7-4485-a43b-735863b3d850" alt="Comparative Result 3" width="400"/>
<img src="blob:https://web.whatsapp.com/5e6a5345-c931-44c2-98b4-dccfe5d6ec60" alt="Comparative Result 3" width="400"/>
<img src="blob:https://web.whatsapp.com/c274ed0f-71df-465a-a8b5-ba9334f97b43" alt="Comparative Result 3" width="400"/>


#### Link to complete result set:
+ [CodeFormer](https://drive.google.com/file/d/1FwWUeVGKWGQP0BgxOEZ8CAmtO-eYAevl/view?usp=sharing)

+ [GFPGAN](https://drive.google.com/file/d/1w6nxfkwU5Z_LU_ekIIrNvQNNa_ioblwk/view?usp=sharing)
