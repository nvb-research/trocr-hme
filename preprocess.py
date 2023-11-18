import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

################################
# First, we remove noise in background and resize to square picture but maintaining its aspect ratio and adding white padding
################################

# Function to resize an image while maintaining its aspect ratio and adding white padding
def resize_with_padding(image, target_size):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling factors for width and height
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Use the smaller scaling factor to ensure the image fits entirely
    scaling_factor = min(width_ratio, height_ratio)

    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a white canvas of the target size
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # Calculate the position to paste the resized image (centered)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Paste the resized image onto the canvas
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image

# Function to remove background noise and convert to black and white
def process_image(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to make it black and white
    _, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)

    return binary_image

# TODO: Define path to the folder containing your images and label
label_file = "label.txt" # Change it to your dir
input_folder = 'images' # Change it to your dir
output_folder = input_folder # Change it if you don't want to replace themselves

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Change the extensions as needed
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        #Resize the image with padding
        target_size = (384, 384)  # Adjust the size as needed
        resized_image = resize_with_padding(image, target_size)

        # Process the image (remove background noise and convert to black and white)
        processed_image = process_image(resized_image)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image)

print("Images processed and saved to the output folder.")

################################
# Then, we need to split dataset into train, test, val forlder with its labels
################################

def label_cs(txt_directory, output_file):

    # Create a list to store the combined lines
    combined_lines = []

    # Loop through each .txt file in the directory
    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            # Read the content of the .txt file
            with open(os.path.join(txt_directory, filename), 'r') as file:
                content = file.read()
                # Combine the filename and content, and add to the list
                combined_line = f"{filename} {content}"
                combined_lines.append(combined_line)

    # Replace ".txt" with ".png" in each line
    combined_lines = [line.replace(".txt", ".png") for line in combined_lines]

    # Write the combined lines to a new .txt file
    with open(output_file, 'w') as file:
        file.write('\n'.join(combined_lines))

    # Delete all files with .txt extension in the specified directory
    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(txt_directory, filename)
            os.remove(file_path)

    print(f"Combined {len(combined_lines)} files into {output_file} successfully!")

# Set the paths to your image folder and label file
image_folder = input_folder

# Load labels from the label file
with open(label_file, 'r') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

# Create a list of image filenames
image_filenames = [f"{i:07d}.png" for i in range(len(labels))]

# Combine image filenames and corresponding labels
data = list(zip(image_filenames, labels))

# Shuffle the data
random.shuffle(data)

# Split the data into train, test, and validation sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Define destination folders for train, test, and validation sets
train_folder = image_folder + "/train/"
test_folder = image_folder + "/test/"
val_folder = image_folder + "/val/"

# Create destination folders if they don't exist
for folder in [train_folder, test_folder, val_folder]:
    os.makedirs(folder, exist_ok=True)

# Move images to their respective folders
for filename, label in train_data:
    os.rename(os.path.join(image_folder, filename), os.path.join(train_folder, filename))
    with open(os.path.join(train_folder, filename.replace(".png", ".txt")), 'w') as f:
        f.write(label)
# Combining Labels for train data
label_cs(train_folder, input_folder + '/labels_train.txt')

for filename, label in test_data:
    os.rename(os.path.join(image_folder, filename), os.path.join(test_folder, filename))
    with open(os.path.join(test_folder, filename.replace(".png", ".txt")), 'w') as f:
        f.write(label)
# Combining Labels for test data
label_cs(test_folder, input_folder + '/labels_test.txt')

for filename, label in val_data:
    os.rename(os.path.join(image_folder, filename), os.path.join(val_folder, filename))
    with open(os.path.join(val_folder, filename.replace(".png", ".txt")), 'w') as f:
        f.write(label)
# Combining Labels for val data
label_cs(val_folder, input_folder + '/labels_val.txt')

print("Data split and moved successfully!")
