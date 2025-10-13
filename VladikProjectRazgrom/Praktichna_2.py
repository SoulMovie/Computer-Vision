import cv2
import numpy as np

hsv = cv2.imread("images/krfoto.jpg")
hsv = cv2.resize(hsv, (500, 500))
img_copy = hsv.copy()

hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
hsv - cv2.GaussianBlur(hsv, (5, 5), 0)

blue_lower = np.array([69, 75, 106])
blue_upper = np.array([132, 255, 255])
blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)


red_lower = np.array([166, 40, 95])
red_upper = np.array([179, 255, 255])
red_mask = cv2.inRange(hsv, red_lower, red_upper)

green_lower = np.array([38, 104, 90])
green_upper = np.array([105, 255, 255])
green_mask = cv2.inRange(hsv, green_lower, green_upper)

black_lower = np.array([0, 0, 0])
black_upper = np.array([179, 255, 71])
black_mask = cv2.inRange(hsv, black_lower, black_upper)

mask_total = cv2.bitwise_or(red_mask, blue_mask)
mask_total = cv2.add(mask_total, black_mask)
mask_total = cv2.add(mask_total, green_mask)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
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
        elif len(approx) > 6:
            shape = "Ouwal"
        else:
            shape = "Shos_inshe"

        cv2.drawContours(img_copy, [approx], -1, (207, 189, 60), 2)
        cv2.circle(img_copy, (cX, cY), 5, (207, 189, 60), -1)
        cv2.putText(img_copy, f'shape: {shape}', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(img_copy, f'A: {int(area)}, P: {int(perimeter)}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)
        cv2.putText(img_copy, f'Center: {cX}, {cY}', (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)






blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in blue_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.putText(img_copy, f'Color: Blue', (x, y -35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in red_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.putText(img_copy, f'Color: Red', (x, y -35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in green_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.putText(img_copy, f'Color: Green', (x, y -35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in black_contours:
    area = cv2.contourArea(cnt)
    if area > 100:

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.putText(img_copy, f'Color: Black', (x, y -35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

cv2.imshow("krfoto", hsv)
cv2.imshow("krfoto", img_copy)
cv2.imshow("blue_mask", blue_mask)
cv2.imshow("red_mask", red_mask)
cv2.imshow("green_mask", green_mask)
cv2.imshow("black_mask", black_mask)
cv2.imshow("mask_total", mask_total)
cv2.waitKey(0)
cv2.destroyAllWindows()