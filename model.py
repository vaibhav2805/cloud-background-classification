import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Define the folder containing your segmented images.
IMAGE_FOLDER = 'augmented_images'

# Set the threshold for classifying as "cloud" or "background."
threshold = 128

# Define the blue color range for cloud detection
lower_blue = np.array([100, 50, 50], dtype=np.uint8)
upper_blue = np.array([140, 255, 255], dtype=np.uint8)


# here change the relative path to get the cloud coverage of any image
sample_image = cv2.imread('augmented_images\label_cloud1_8_%cloud_91_%background%augmented_0.jpg', cv2.IMREAD_COLOR) 


def preprocess_image(image):
    if image is not None:
        # Create a mask for the blue regions in the image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Create "cloud" and "background" images based on the mask
        cloud = cv2.bitwise_and(image, image, mask=mask)
        background = cv2.bitwise_and(image, image, mask=1 - mask)

        return cloud, background
    else:
        return None, None

def calculate_coverage(cloud_image, background_image):
    total_pixels = cloud_image.size + background_image.size
    cloud_coverage = (cloud_image.size / total_pixels) * 100
    background_coverage = 100 - cloud_coverage
    return cloud_coverage, background_coverage

# Load and preprocess the dataset
cloud_images = []
background_images = []

for filename in os.listdir(IMAGE_FOLDER):
    image = cv2.imread(os.path.join(IMAGE_FOLDER, filename), cv2.IMREAD_COLOR)

    cloud_image, background_image = preprocess_image(image)

    if cloud_image is not None and background_image is not None:
        cloud_images.append(cloud_image.flatten())
        background_images.append(background_image.flatten())

# Create labels for cloud and background (corrected)
labels = [0] * len(cloud_images) + [1] * len(background_images)  # Swap the labels for cloud and background

# Combine the data for training
X = cloud_images + background_images

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create an SVM classifier with a custom C value
custom_C = 1.0  # You can adjust this value as needed
clf = SVC(C=custom_C, kernel='linear')

# Train the SVM model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print('Accuracy:', accuracy)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


if sample_image is not None:
    # Create a mask for the blue regions in the sample image
    hsv_sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
    sample_mask = cv2.inRange(hsv_sample_image, lower_blue, upper_blue)

    # Calculate the percentage coverage of each class based on the number of pixels
    cloud_pixels = np.sum(sample_mask)  # Count the number of "cloud" pixels in the mask
    total_pixels = sample_image.size  # Total number of pixels in the image

    cloud_coverage = (cloud_pixels / total_pixels) * 100
    background_coverage = 100 - cloud_coverage

    # Print the results
    print('Cloud coverage:', cloud_coverage, '%')
    print('Background coverage:', background_coverage, '%')
    print(sample_image)
else:
    print('Sample image not loaded or invalid.')
