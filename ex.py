# 두 사진의 유사성을 수치로 나타내줌

# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3
img1 = cv2.imread("park.jpg")
img2 = cv2.imread("yoo.jpg")

# STEP 4
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

# print(faces1[0])

# STEP 5
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./park_output.jpg", rimg)


feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sim = np.dot(feat1, feat2.T)
print(sim)

