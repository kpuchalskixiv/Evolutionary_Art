import cv2
import numpy as np


canvas_width, canvas_height = 571, 356
target_canvas = np.ones((canvas_height, canvas_width, 3))

mona_lisa = cv2.imread("mona.jpg")
percent = 256 / mona_lisa.shape[0]
width = int(np.floor(mona_lisa.shape[1] * percent))
height = int(mona_lisa.shape[0] * percent)
mona_lisa = cv2.resize(mona_lisa, (width, height), interpolation = cv2.INTER_AREA)

height, width, _ = mona_lisa.shape

target_canvas[50:50+height, 320:320+width, :] = mona_lisa
target_canvas = cv2.putText(target_canvas, "target image", (350, 336), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=0)
target_canvas = cv2.putText(target_canvas, "current solution", (95, 336), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=0)

step = 300
iteration = 0
phenotypes = np.load("best_phenotypes.npy")

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter("clip.avi", fourcc, 20, (canvas_width*2, canvas_height*2), isColor=True)
best = np.load("best_image.npy")
phenotypes[-1, ...] = best

for phenotype in phenotypes:
    target_canvas[50:50+height, 80:80+width, :] = phenotype[:,:,::-1]
    video_canvas = cv2.putText(target_canvas.copy(), f"Iteration: {iteration}", (225, 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=0)
    video_canvas = cv2.resize(video_canvas, (canvas_width*2, canvas_height*2), interpolation=cv2.INTER_AREA)
    out.write(video_canvas.astype(np.uint8))
    iteration += step
cv2.destroyAllWindows() 
out.release()
