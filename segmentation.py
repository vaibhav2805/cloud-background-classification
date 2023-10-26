import os
import cv2
import numpy as np

def extract_image_features(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average color of the sky (blue)
    average_sky_color = np.mean(image[0:100, 0:100], axis=(0, 1))

    # Apply Canny edge detection to detect edges in the image
    edges = cv2.Canny(gray, threshold1=30, threshold2=120)

    # Calculate the total number of edge pixels
    total_edge_pixels = np.sum(edges > 0)

    # Calculate the percentage of edge pixels in the image
    edge_pixel_percentage = (total_edge_pixels / (image.shape[0] * image.shape[1])) * 100

    # Extract additional features as needed, such as texture, shape, or cloud cover percentage

    # For example, you could use color segmentation to estimate cloud cover percentage

    # Define regions with blue color (sky)
    lower_blue = np.array([90, 100, 100], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    # Create a mask for the blue regions
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Calculate the cloud cover percentage
    cloud_cover_percentage = (np.sum(mask > 0) / (image.shape[0] * image.shape[1])) * 100

    # Return the extracted features as a dictionary
    features = {
        "average_sky_color": average_sky_color,
        "edge_pixel_percentage": edge_pixel_percentage,
        "cloud_cover_percentage": cloud_cover_percentage,
    }

    return features

# Define the folder containing the images
image_folder = "raw-frames"

# Create the output directory if it doesn't exist
output_folder = "segmented_images"
os.makedirs(output_folder, exist_ok=True)

# Iterate through the images in the folder
i=0
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg')):
        image_path = os.path.join(image_folder, filename)
        image_features = extract_image_features(image_path)

        # Read the image for segmentation
        image = cv2.imread(image_path)

        # Perform segmentation (for example, using color segmentation)
        lower_blue = np.array([90, 100, 100], dtype=np.uint8)
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(image, lower_blue, upper_blue)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        # Extract the cloud cover percentage from features
        cloud_cover_percentage = int(image_features["cloud_cover_percentage"])

        # Construct the new filename with label and percentage of cloud
        new_filename = f"label_cloud{i}_{cloud_cover_percentage}.jpg"
        i=i+1

        # Save the segmented image in the output directory with the new filename
        output_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(output_path, segmented_image)

# Print a message indicating the completion
print("Segmented images saved in the 'segmented_images' folder with label and percentage of cloud as names.")

