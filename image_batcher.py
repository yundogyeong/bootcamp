import os
import sys
import numpy as np
import random
from PIL import Image

def resize_to_target(image, im_width=32, im_height=32):
    scale_width = im_width / image.width
    scale_height = im_height / image.height
    scale = min(scale_width, scale_height)
    resized_image = image.resize((im_width, im_height), resample=Image.BILINEAR)
    return resized_image

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert(mode='RGB')
    image = resize_to_target(image, 32, 32)
    
    # Adding a simple image augmentation step: 
    # For example : horizontal flip
    # image = ImageOps.mirror(image)
    
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))

    return image

class ImageBatcher:
    def __init__(self, input, shape, dtype, max_num_images=None, exact_batches=False, preprocessor=preprocess_image):

        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
        
        if os.path.isdir(input):
            for root, _, files in os.walk(input):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_image(file_path):
                        self.images.append(file_path)
            self.images.sort()

        #Shuffle calibration image
        random.shuffle(self.images)
        self.num_images = len(self.images)
        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        self.batch_size = shape[0]
        self.height = self.shape[2]
        self.width = self.shape[3]
 
        # Adapt the number of images as needed
        self.num_images = max_num_images
        self.num_images = self.batch_size * (self.num_images // self.batch_size)

        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor

    def get_batch(self):
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            #batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = self.preprocessor(image)
            self.batch_index += 1
            yield batch_data, batch_images
