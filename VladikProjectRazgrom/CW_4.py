import cv2
import numpy as np

img = cv2.imread("images/hogridar.jpg")
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img - cv2.GaussianBlur(img, (5, 5), 0)

lower = np.array([2, 0, 23])
upper = np.array([86, 255, 255])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round(4 * np.pi * area / (perimeter) ** 2, 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = "Trikutnik"
        elif len(approx) == 4:
            shape = "Qwudratik"
        elif len(approx) > 8:
            shape = "Ouwal"
        else:
            shape = "Shos_inshe"

        cv2.drawContours(img_copy, [approx], -1, (0, 0, 0), 2)
        cv2.circle(img_copy, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(img_copy, f'shape: {shape}', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(img_copy, f'A: {int(area)}, P: {int(perimeter)}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img_copy, f'AR: {aspect_ratio}, C: {compactness}', (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)




cv2.imshow("Original", mask)
cv2.imshow("Trippy", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()