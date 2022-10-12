similarity_threshold = .02

fps = 30
patch_size = 126, 222  # width, height
fovs = 115, 90  # horizontal, vertical, in degrees
central_fov = 13  # fov of the fovea
near_peripheral_fov = 30  # physio fov
mid_perpheral_fov = 60  # physio fov

# for drawing the fovea box
patch_color = (255, 255, 0)
center_color = (255, 0, 0)
fovea_color = (0, 255, 0)
parafovea_color = (0, 0, 255)
peripheri_color = (0, 255, 255)

image_shape = (800, 800, 3)
image_size = image_shape[0], image_shape[1]
ppds = image_size[0] / fovs[0], image_size[1] / fovs[1]  # horizontal, vertical, calculate pixel per degree of FOV