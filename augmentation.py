import os
import cv2

# Define the input folder containing the segmented images
segmented_image_folder = "segmented_images"

# Create the output directory for augmented images if it doesn't exist
augmented_image_folder = "augmented_images"
os.makedirs(augmented_image_folder, exist_ok=True)

# Function to apply image augmentation
def augment_image(image):
    # Define the augmentation operations you want to apply here.
    # For example, you can apply operations like rotation, flipping, brightness adjustment, etc.
    # Here, we'll demonstrate rotation and horizontal flipping as examples.

    # Rotate the image (you can specify the angle)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Horizontal flip the image
    flipped_image = cv2.flip(image, 1)

    return [rotated_image, flipped_image]

# Iterate through the segmented images in the folder
for filename in os.listdir(segmented_image_folder):
    if filename.endswith('.jpg'):
        segmented_image_path = os.path.join(segmented_image_folder, filename)

        # Read the segmented image
        segmented_image = cv2.imread(segmented_image_path)

        # Augment the image
        augmented_images = augment_image(segmented_image)

        # Save augmented images with new filenames
        for i, augmented_image in enumerate(augmented_images):
            augmented_filename = f"{filename.replace('.jpg', '')}%augmented_{i}.jpg"
            augmented_image_path = os.path.join(augmented_image_folder, augmented_filename)
            cv2.imwrite(augmented_image_path, augmented_image)

# Print a message indicating the completion
print("Augmented images saved in the 'augmented_images' folder.")
