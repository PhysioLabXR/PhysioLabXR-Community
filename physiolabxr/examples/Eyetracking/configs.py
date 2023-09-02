fixation_min_duration = .15
fps = 30
fixation_min_frame_count = int(fixation_min_duration * fps)
similarity_threshold = .04

patch_size = 63, 111  # width, height
fovs = 115, 90  # horizontal, vertical, in degrees
central_fov = 13  # fov of the fovea in degrees
near_peripheral_fov = 30  # physio fov
mid_perpheral_fov = 60  # physio fov

# for drawing the fovea box
patch_color = (255, 255, 0)
center_color = (255, 0, 0)
fovea_color = (0, 255, 0)
parafovea_color = (0, 0, 255)
peripheri_color = (0, 255, 255)

image_shape = (400, 400, 3)
image_size = image_shape[0], image_shape[1]
ppds = image_size[0] / fovs[0], image_size[1] / fovs[1]  # horizontal, vertical, calculate pixel per degree of FOV