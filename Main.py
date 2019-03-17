from ImageGenerator import *
import os

image_generator = ImageGenerator(backgrounds_folder='Backgrounds',
                                 fonts_folder='Fonts',
                                 symbols_folder='Symbols')

image_generator.print_directory_status()

# image_generator.generate_text_image(os.path.join(image_generator.fonts_folder, 'FilamentFive.otf'), 40, 'LUKAS')

image_generator.generate_examples_for_all_symbols(15)

# image_generator.combine_images(smaller_image_location=os.path.join(image_generator.current_directory,
#                                                                    't.png'),
#                                background_image_location=os.path.join(image_generator.backgrounds_folder,
#                                                                       'background_1.jpg'))
