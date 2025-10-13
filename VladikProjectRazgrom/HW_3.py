import cv2
import numpy as np

img = cv2.imread("images/simon.jpg")
img = cv2.resize(img, (400, 400))

cv2.rectangle(img, (140, 0), (270, 130), (255, 0, 255), 2)
cv2.putText(img, "Vladig Govadovskii", (85, 155), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 0, 255))

cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()