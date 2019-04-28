from ImageGenerator import *
import time


image_generator = ImageGenerator(backgrounds_folder='Backgrounds',
                                 fonts_folder='Fonts',
                                 symbols_folder='Symbols')

image_generator.print_directory_status()

# Step 1
# image_generator.generate_examples_for_all_symbols(len(os.listdir(image_generator.fonts_folder)))

# s_locations = [os.path.join(image_generator.symbols_folder, '//Symbol_/_11.png')]
#
# Step 2
# image_generator.combine_images(smaller_images_locations=s_locations,
#                                background_image_location=os.path.join(image_generator.backgrounds_folder,
#                                                                       'A4_math_1.png'))

times = list()
# Step 3 (For testing after a training session)
for i in range(10):
    start_time = time.time()
    image_generator.start_generating_images(1, 10, 'Easy')
# image_generator.generate_math_expression('2+2')
    end_time = time.time()
    times.append((end_time - start_time))
time_sum = 0
for item in times:
    time_sum += item

print('Average time for execution:', time_sum / 10)
