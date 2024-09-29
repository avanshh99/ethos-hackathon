# Round 1 solution for ML statement of Ethos Hackathon

## Approach:
 1) Frame Extraction from Videos:
    Frames are extracted from the video using libraries like OpenCV or FFmpeg. This step captures individual frames at regular intervals to isolate images for further processing.

 2) Frames Sent to GAN Model (Enhancement):
    The extracted frames are passed through enhancement models like CodeFormer, Real-ESRGAN, or BSRGAN. These GAN-based models restore image quality, especially for faces in low-resolution or blurry frames.

 3) Preprocessing Steps:
    Face Detection: Models such as MTCNN (Multi-task Cascaded Convolutional Networks) or YOLOv5 detect and crop faces from each frame, isolating the face for further steps.
    Alignment and Normalization: Detected faces are aligned and normalized, preparing them for facial reconstruction.

  4) Facial Reconstruction (Deep Learning):
    CodeFormer or GFPGAN is used for facial reconstruction, enhancing facial features and ensuring realistic results.
    DeepFace (which may utilize models like Facenet or ArcFace) can extract facial features, while AutoEncoders or Variational AutoEncoders (VAEs) can regenerate high-quality facial images from these features.

## Result
### Extracted Frame
![Extracted Frame](https://cdn.discordapp.com/attachments/1284411927115595786/1289821731502161962/frame_0_1.jpg?ex=66fa3798&is=66f8e618&hm=98147268bab760d8c16d128d9785b3bf6224fe15ac94aedcf97c17b4a411a8ee&)

### Comparitive result
![Comparitive Reult](https://i.postimg.cc/2j4jH3CT/poc-submission.png)

