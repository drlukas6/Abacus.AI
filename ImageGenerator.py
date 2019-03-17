import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self, backgrounds_folder, fonts_folder, symbols_folder):
        self.current_directory = os.getcwd()
        self.backgrounds_folder = os.path.join(self.current_directory, backgrounds_folder)
        if not os.path.exists(backgrounds_folder):
            os.mkdir(backgrounds_folder)
        self.fonts_folder = os.path.join(self.current_directory, fonts_folder)
        if not os.path.exists(fonts_folder):
            os.mkdir(fonts_folder)
        self.symbols_folder = os.path.join(self.current_directory, symbols_folder)
        if not os.path.exists(symbols_folder):
            os.mkdir(symbols_folder)
        self.available_fonts = os.listdir(self.fonts_folder)
        self.symbols = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/'}

    def print_directory_status(self):
        print('Current directory: {}'.format(self.current_directory))
        print('Backgrounds directory: {}'.format(self.backgrounds_folder))
        print('Fonts directory: {}'.format(self.fonts_folder))
        print('Symbols directory: {}'.format(self.symbols_folder))

    def generate_text_image(self, font, size, text):
        try:
            loaded_font = ImageFont.truetype(font, size)
        except:
            print('FONT: ', font)
        size = loaded_font.getsize(text)
        background_image = Image.new('RGBA', size, (255, 255, 255, 0))
        image_to_draw_on = ImageDraw.Draw(background_image)
        image_to_draw_on.text((0, 0), text, (0, 0, 0), loaded_font)
        return background_image

    def generate_examples_for_text(self, text, number_of_examples):
        for i in range(0, number_of_examples):
            symbol_directory = os.path.join(self.symbols_folder, text)
            if not os.path.exists(symbol_directory):
                os.mkdir(symbol_directory)
            random_font_size = np.random.randint(50, 70)
            random_font_index = np.random.randint(0, len(self.available_fonts))
            random_font_location = os.path.join(self.fonts_folder,
                                            self.available_fonts[random_font_index])
            generated_image = self.generate_text_image(random_font_location,
                                                       random_font_size,
                                                       text=text)
            image_path = os.path.join(symbol_directory, '{}_{}.png'.format(text, i))
            generated_image.save(image_path, format='PNG')

    def generate_examples_for_all_symbols(self, number_of_examples):
        for symbol in self.symbols:
            self.generate_examples_for_text(symbol, number_of_examples)

    def combine_images(self, smaller_image_location, background_image_location):
        smaller_image = cv2.imread(smaller_image_location, -1)
        s_height, s_width, s_channels = smaller_image.shape
        s_alpha = smaller_image[:, :, 3] / 255.0
        l_alpha = 1.0 - s_alpha
        background_image = cv2.imread(background_image_location)
        b_height, b_width, b_channels = background_image.shape
        x_offset = np.random.randint(0, (b_width - s_width))
        x_max = x_offset + s_width
        y_offset = np.random.randint(0, (b_height - s_height))
        y_max = y_offset + s_height
        background_copy = background_image.copy()
        for c in range(0, 3):
            background_copy[y_offset:y_max, x_offset:x_max, c] = (s_alpha * smaller_image[:, :, c] +
                                                                  l_alpha * background_copy[y_offset:y_max, x_offset:x_max, c])
        plt.imshow(background_copy)
        plt.show()
