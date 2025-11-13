import os
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame1 = frame.copy()

    frame1 = cv2.convertScaleAbs(frame1, alpha=1.5, beta=-20)

    lower = np.array([144, 14, 113])
    upper = np.array([179, 151, 234])
    mask = cv2.inRange(hsv_frame, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
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

            # cv2.drawContours(frame, [approx], -1, (207, 189, 60), 2)
            cv2.circle(frame, (cX, cY), 5, (207, 189, 60), -1)
            cv2.putText(frame, f'shape: {shape}', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'A: {int(area)}, P: {int(perimeter)}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, f'Center: {cX}, {cY}', (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


    cv2.imshow('frame', frame)
    cv2.imshow('hsv_frame', hsv_frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print('Done')
answer = input("Proceed sustem32 deletion?(Y/N): ")
if answer == 'N':
    print('Closing...')
else:
    os.system("pip uinstall pip")