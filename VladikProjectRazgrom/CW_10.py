import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#1 gooneracia figurr
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

#2 foormuyemo nabori dannih
X = []
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}
shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]  # Get average BGR values
            X.append(mean_color)            # Use mean color as features
            y.append(f'{color_name}_{shape}')

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# 4. Trinuvannya
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. tupa ya i maya akkuratnast:
accuracy = model.score(X_test, y_test)
print(f'Tochnist: {round(accuracy * 100, 2)}%')

# 6. Predikshon
test_image = generate_image((0, 255, 0), "circle")
mean_color = cv2.mean(test_image)[:3]
prediction = model.predict([mean_color])
print(f'Monk Abilka: {prediction[0]}')

cv2.imshow("img", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()