import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


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
        self.symbols = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', ':'}

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
            random_font_index = np.random.randint(0, len(self.available_fonts))
            random_font_location = os.path.join(self.fonts_folder, self.available_fonts[random_font_index])
            if '.DS_Store' in random_font_location:
                continue
            random_font_size = np.random.randint(70, 100)
            generated_image = self.generate_text_image(font=random_font_location,
                                                       size=random_font_size,
                                                       text=text)
            image_path = os.path.join(symbol_directory, 'Symbol_{}_{}.png'.format(text, i))
            generated_image.save(image_path, format='PNG')

    def generate_examples_for_all_symbols(self, number_of_examples):
        for symbol in self.symbols:
            print('Processing symbol: ', symbol, ' ...')
            self.generate_examples_for_text(symbol, number_of_examples)

    def combine_images(self, smaller_images_locations, background_image_location):
        background_image = cv2.imread(background_image_location)
        b_height, b_width, b_channels = background_image.shape

        background_copy = background_image.copy()

        for s_location in smaller_images_locations:
            smaller_image = cv2.imread(s_location, -1)
            # Adding 0.7 so the image never gets smaller that .7 of its size
            random_scaling_index_x = np.random.rand() + 0.7
            random_scaling_index_y = np.random.rand() + 0.7
            smaller_image = cv2.resize(smaller_image,
                                      None,
                                      fx=random_scaling_index_x,
                                      fy=random_scaling_index_y)
            s_height, s_width, s_channels = smaller_image.shape
            s_height_3, s_width_3 = math.floor(s_height / 3), math.floor(s_width / 3)

            affine_ref_points_1 = np.float32([[0, 0],
                                              [s_height, 0],
                                              [0, s_width]])

            affine_ref_points_2 = np.float32([[self.random_float(0, s_height_3), self.random_float(0, s_width_3)],
                                              [self.random_float(s_height - s_height_3, s_height), self.random_float(0, s_width_3)],
                                              [self.random_float(0, s_height_3), self.random_float(s_width - s_width_3, s_width)]])

            M = cv2.getAffineTransform(affine_ref_points_1, affine_ref_points_2)

            smaller_image = cv2.warpAffine(smaller_image, M, (s_width, s_height))

            s_alpha = smaller_image[:, :, 3] / 255.0
            l_alpha = 1.0 - s_alpha

            x_offset = np.random.randint(0, (b_width - s_width))
            x_max = x_offset + s_width
            y_offset = np.random.randint(0, (b_height - s_height))
            y_max = y_offset + s_height
            for c in range(0, 3):
                background_copy[y_offset:y_max, x_offset:x_max, c] = (s_alpha * smaller_image[:, :, c] +
                                                                      l_alpha * background_copy[y_offset:y_max, x_offset:x_max, c])
        plt.imshow(background_copy)
        plt.show()

    def random_float(self, min_num, max_num):
        return np.random.random() * (max_num - min_num) + min_num
