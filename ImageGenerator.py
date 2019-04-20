import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import math
import random
import string
import csv


class ImageGenerator:
    """
    Class that defines a pipeline needed for:
        - Creating and maintaining a well defined folder structure
        - Creating examples of symbols needed for Object Detection in various fonts
        - Performing geometric transformations on symbols before pasting them for more unique images
        - Pasting symbols on various and random backgrounds
        - Saving images and maintaining *.csv files for further usage in the pipeline
        - Skipping *.xml files completely makes the whole process more streamlined
    """

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
        self.generated_images_folder = os.path.join(self.current_directory, 'Images')
        self.images_folder = os.path.join(self.current_directory, 'Images')
        if not os.path.exists(self.images_folder):
            os.mkdir(self.images_folder)
        self.data_folder = os.path.join(self.current_directory, 'TrainData')
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        self.expressions_folder = os.path.join(self.current_directory, 'Expressions')
        if not os.path.exists(self.expressions_folder):
            os.mkdir(self.expressions_folder)

    def print_directory_status(self):
        print('Current directory: {}'.format(self.current_directory))
        print('Backgrounds directory: {}'.format(self.backgrounds_folder))
        print('Fonts directory: {}'.format(self.fonts_folder))
        print('Symbols directory: {}'.format(self.symbols_folder))

    def generate_text_image(self, font, size, text):
        """
        :param font: Font to write text in
        :param size: Size of text
        :param text: Text to write on a transparent image
        :return: Created image in Pil.Image format
        """
        loaded_font = ImageFont.truetype(font, size)
        size = loaded_font.getsize(text)
        background_image = Image.new('RGBA', size, (255, 255, 255, 0))
        image_to_draw_on = ImageDraw.Draw(background_image)
        image_to_draw_on.text((0, 0), text, (0, 0, 0), loaded_font)
        return background_image

    def generate_examples_for_text(self, text, number_of_examples):
        """
        :param text: Text to write on a transparent image
        :param number_of_examples: Number of images to generate
        :return: /
        """
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

    def generate_math_expression(self, expression, font_path='Fonts/JustBreathe.otf',
                                 size=100, background_location='Backgrounds/A4_math_2.png'):
        """
        :param expression: Math expression (eg. 32+5) to write on a transparent image
        :param font_path: Path to a font to use for writing text
        :param size: Size of text
        :param background_location: Location of a background image to paste text on
        :return: /
        """
        print('Entered expression:', expression)

        full_font_path = os.path.join(self.current_directory, font_path)
        print('Full font path:', full_font_path)

        generated_expression = self.generate_text_image(full_font_path, size, expression)
        image_path = os.path.join(self.current_directory, 'test.png')
        generated_expression.save(image_path, format='PNG')

        background_image_location = os.path.join(self.current_directory, background_location)
        background_image = cv2.imread(background_image_location)

        b_height, b_width, b_channels = background_image.shape
        background_copy = background_image.copy()
        foreground = cv2.imread(image_path, -1)
        s_height, s_width, s_channels = foreground.shape

        s_alpha = foreground[:, :, 3] / 255.0
        l_alpha = 1.0 - s_alpha

        x_offset = int((b_width / 2) - (s_width / 2))  # Centering the expression horizontally
        x_max = int(x_offset + s_width)
        y_offset = int((b_height / 2) - (s_height / 2))  # Centering the expression vertically
        y_max = int(y_offset + s_height)

        for c in range(0, 3):
            background_copy[y_offset:y_max, x_offset:x_max, c] = (s_alpha * foreground[:, :, c] +
                                                                  l_alpha * background_copy[y_offset:y_max,
                                                                  x_offset:x_max, c])
        image_name = '{}.png'.format(self.random_word(6))
        cv2.imwrite(os.path.join(self.expressions_folder, image_name), background_copy)

    def generate_examples_for_all_symbols(self, number_of_examples):
        """
        :param number_of_examples: How many images to generate for each symbol
        :return: /
        """
        for symbol in self.symbols:
            print('Processing symbol: ', symbol, ' ...')
            self.generate_examples_for_text(symbol, number_of_examples)

    def combine_images(self, smaller_images_locations, background_image_location):
        """
        :param smaller_images_locations: Full path locations to smaller images to use as foreground
        :param background_image_location: Background to paste foreground on
        :return: Line of *.csv file describing the image and its content
        """
        print('Smaller images: {0} ; Background: {1}'.format(smaller_images_locations, background_image_location))

        background_image = cv2.imread(background_image_location)
        b_height, b_width, b_channels = background_image.shape

        background_copy = background_image.copy()
        image_class = ''
        for s_location in smaller_images_locations:
            smaller_image_location = os.path.join(self.symbols_folder, s_location)
            smaller_image = cv2.imread(smaller_image_location, -1)
            # Adding 0.7 so the image never gets smaller that 1x of its size
            random_scaling_index_x = np.random.rand() + 1
            random_scaling_index_y = np.random.rand() + 1
            smaller_image = cv2.resize(smaller_image,
                                       None,
                                       fx=random_scaling_index_x,
                                       fy=random_scaling_index_y)
            s_height, s_width, s_channels = smaller_image.shape
            s_height_3, s_width_3 = math.floor(s_height / 5), math.floor(s_width / 4)

            affine_ref_points_1 = np.float32([[0, 0],
                                              [s_height, 0],
                                              [0, s_width]])

            affine_ref_points_2 = np.float32([[self.random_float(0, s_height_3), self.random_float(0, s_width_3)],
                                              [self.random_float(s_height - s_height_3, s_height),
                                               self.random_float(0, s_width_3)],
                                              [self.random_float(0, s_height_3),
                                               self.random_float(s_width - s_width_3, s_width)]])

            M = cv2.getAffineTransform(affine_ref_points_1, affine_ref_points_2)

            smaller_image = cv2.warpAffine(smaller_image, M, (s_width, s_height))

            s_alpha = smaller_image[:, :, 3] / 255.0
            l_alpha = 1.0 - s_alpha

            x_offset = np.random.randint(0, (b_width - s_width - 10))
            x_max = x_offset + s_width
            y_offset = np.random.randint(0, (b_height - s_height - 10))
            y_max = y_offset + s_height
            for c in range(0, 3):
                background_copy[y_offset:y_max, x_offset:x_max, c] = (s_alpha * smaller_image[:, :, c] +
                                                                      l_alpha * background_copy[y_offset:y_max,
                                                                                x_offset:x_max, c])
        # class imagename width height xmin xmax ymin ymax
            image_class += s_location[0]
        image_name = '{}.png'.format(self.random_word(6))
        cv2.imwrite('{0}/{1}'.format(self.images_folder, image_name), background_copy)
        csv_line = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'.format(image_name, image_class, b_width, b_height, x_offset, x_max, y_offset, y_max)
        return csv_line

    def start_generating_images(self, images_per_background, number_of_images, difficulty):
        """
        :param images_per_background: How many symbols to paste on a background
        :param number_of_images: Number of images to generate
        :param difficulty: Either Easy or Challenging
        :return: Creates a csv file
        """
        csv_lines = list()
        for i in range(number_of_images):
            random_images = []
            while len(random_images) < images_per_background:
                random_letter = os.listdir(self.symbols_folder)[np.random.randint(0, len(os.listdir(self.symbols_folder)))]
                if random_letter == '.DS_Store':
                    continue
                random_letter_folder = os.path.join(self.symbols_folder, random_letter, difficulty)
                found_images = os.listdir(random_letter_folder)
                random_image_name = found_images[np.random.randint(0, len(found_images))]
                random_image_path = os.path.join(random_letter, difficulty, random_image_name)
                if '.DS_S' in random_image_path:
                    continue
                random_images.append(random_image_path)
            while True:
                random_background_location = os.path.join(self.backgrounds_folder,
                                                          os.listdir(self.backgrounds_folder)[np.random.randint(0, len(os.listdir(self.backgrounds_folder)))])
                if '.DS_Store' not in random_background_location:
                    break
            csv_line = self.combine_images(random_images, random_background_location)
            csv_lines.append(csv_line)
            csv_train_name = 'train01.csv'
            csv_test_name = 'test01.csv'
            csv_train_location = os.path.join(self.data_folder, csv_train_name)
            csv_test_location = os.path.join(self.data_folder, csv_test_name)
            train_lines = csv_lines[0:int(len(csv_lines) * 0.8)]
            test_lines = csv_lines[int(len(csv_lines) * 0.8):]

            # train
            with open(csv_train_location, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow('filename, class, imwidth, imheight, xmin, xmax, ymin, ymax'.split(', '))
                for line in train_lines:
                    splitline = line.split(', ')
                    filewriter.writerow(splitline)

            # test
            with open(csv_test_location, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow('filename, class, imwidth, imheight, xmin, xmax, ymin, ymax'.split(', '))
                for line in test_lines:
                    splitline = line.split(', ')
                    filewriter.writerow(splitline)

    @staticmethod
    def random_float(min_num, max_num):
        return np.random.random() * (max_num - min_num) + min_num

    @staticmethod
    def random_word(length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
