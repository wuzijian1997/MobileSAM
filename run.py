import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

img_path = '/home/zijian/Codes/segment-anything/images/frame0000.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# # INIT LOGGERS
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings=np.zeros((repetitions,1))

predictor = SamPredictor(mobile_sam)
predictor.set_image(img)

input_point = np.array([[632, 520], [1211, 539]])
input_label = np.array([1, 1])

# starter.record()
masks, scores, _ = predictor.predict(
    point_coords = input_point,
    point_labels = input_label,
)
# ender.record()
# torch.cuda.synchronize()
# curr_time = starter.elapsed_time(ender)
# print(curr_time)

max_score = scores.max()
max_idx = np.where(scores == max_score)
mask = masks[max_idx]
mask_img = mask.astype(np.uint8).squeeze()
mask_img =  mask_img * 255

# plt.imshow(mask_img)
# plt.show()