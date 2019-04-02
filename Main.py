from ImageGenerator import *
import os

image_generator = ImageGenerator(backgrounds_folder='Backgrounds',
                                 fonts_folder='Fonts',
                                 symbols_folder='Symbols')

image_generator.print_directory_status()

# image_generator.generate_examples_for_all_symbols(len(os.listdir(image_generator.fonts_folder)))

# s_locations = [os.path.join(image_generator.symbols_folder, '//Symbol_/_11.png')]
#
# image_generator.combine_images(smaller_images_locations=s_locations,
#                                background_image_location=os.path.join(image_generator.backgrounds_folder,
#                                                                       'A4_math_1.png'))

image_generator.start_generating_images(False, 1, 100, 'Easy')
