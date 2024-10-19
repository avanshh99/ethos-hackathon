from worker import image_matching_workflow, process_video_workflow

res = image_matching_workflow(img_path="hughJackman.png", base_directory="Celebrity Faces Dataset", output_directory='MATCHES')
print(res)
res_2 = process_video_workflow(video_path='content\circus.mp4', output_dir="output_frames")
print(res_2)