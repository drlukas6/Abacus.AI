# from utils import label_map_util
# from utils import visualization_utils as vis_util
# from utils import ops as utils_ops
# import numpy as np
# import os
import sys
# import tensorflow as tf
#
# from matplotlib import pyplot as plt
# from PIL import Image

from TensorflowUtilities import TensorflowUtilities

# def load_image_into_numpy_array(image):
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape(
#         (im_height, im_width, 3)).astype(np.uint8)
#
# def predict_for_image(image, graph):
#     with graph.as_default():
#         with tf.Session() as sess:
#             ops = tf.get_default_graph().get_operations()
#             all_tensor_names = {output.name for op in ops for output in op.outputs}
#             tensor_dict = {}
#             for key in [
#                 'num_detections', 'detection_boxes', 'detection_scores',
#                 'detection_classes', 'detection_masks'
#             ]:
#                 tensor_name = key + ':0'
#                 if tensor_name in all_tensor_names:
#                     tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#                         tensor_name)
#             if 'detection_masks' in tensor_dict:
#                 # The following processing is only for single image
#                 detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#                 detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#                 # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#                 real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#                 detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#                 detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#                 detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#                     detection_masks, detection_boxes, image.shape[1], image.shape[2])
#                 detection_masks_reframed = tf.cast(
#                     tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#                 # Follow the convention by adding back the batch dimension
#                 tensor_dict['detection_masks'] = tf.expand_dims(
#                     detection_masks_reframed, 0)
#             image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
#
#             # Run inference
#             output_dict = sess.run(tensor_dict,
#                                    feed_dict={image_tensor: image})
#
#             # all outputs are float32 numpy arrays, so convert types as appropriate
#             output_dict['num_detections'] = int(output_dict['num_detections'][0])
#             output_dict['detection_classes'] = output_dict[
#                 'detection_classes'][0].astype(np.uint8)
#             output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#             output_dict['detection_scores'] = output_dict['detection_scores'][0]
#             if 'detection_masks' in output_dict:
#                 output_dict['detection_masks'] = output_dict['detection_masks'][0]
#         return output_dict
#
#
# symbol_mappings = {
#                   1: '0',
#                   2: '1',
#                   3: '2',
#                   4: '3',
#                   5: '4',
#                   6: '5',
#                   7: '6',
#                   8: '7',
#                   9: '8',
#                   10: '9',
#                   11: '+',
#                   12: '-',
#                   13: '*',
#                   14: ':'}
#
# current_dir = os.getcwd()
# model_name = 'final_run_ssd'
# frozen_graph_path = 'models/research/object_detection/final_run_ssd/frozen_inference_graph.pb'
# labels_path = 'models/research/object_detection/training/label_map.pbtxt'
#
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#
# category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)
#
# images_dir = os.path.join(current_dir, 'Images')
# images_to_test = [
#                     # os.path.join(images_dir, 'fiveplusone.jpg'),
#                     # os.path.join(images_dir, 'threeplusseven.jpg'),
#                     # os.path.join(images_dir, 'handwriten_one.jpg'),
#                     os.path.join(images_dir, 'hand_two.jpg'),
#                     os.path.join(images_dir, 'hand_three.jpg'),
#                 ]
#
# for path in images_to_test:
#     print('Image path:', path)
#
# output_size = (24, 16)
#
# image_count = 1
# for image_path in images_to_test:
#     image = Image.open(image_path)
#     # the array based representation of the image will be used later in order to prepare the
#     # result image with boxes and labels on it.
#     image_np = load_image_into_numpy_array(image)
#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#     # Actual detection.
#     output_dict = predict_for_image(image_np_expanded, detection_graph)
#     coordinates = output_dict['detection_boxes']
#
#     # Sorting findings by x-coordinates
#     filtered_xmins = list()
#     for box in coordinates:
#         if box[0] == 0.0 and box[1] == 0.0 and box[2] == 0.0 and box[3] == 0.0:
#             continue
#         filtered_xmins.append(box[1])
#     found_classes = output_dict['detection_classes']
#     filtered_classes = list()
#     for found_class in found_classes:
#         if found_class == 1:
#             continue
#         filtered_classes.append(found_class)
#
#     filtered_classes = [x for _, x in sorted(zip(filtered_xmins, filtered_classes))]
#     print(filtered_classes)
#     found_string = ''
#     for key in filtered_classes:
#         found_string += symbol_mappings[key]
#     print('found string:', found_string)
#     result = ''
#     try:
#         result = eval(found_string)
#     except:
#         result = 'Evaluating failed'
#     print('Result:', result)
#     # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks'),
#         use_normalized_coordinates=True,
#         line_thickness=4)
#     plt.figure(figsize=output_size)
#     plt.imshow(image_np)
#     plt.savefig('output_{}.png'.format(image_count))
#     image_count += 1

arguments = sys.argv
arguments = arguments[1:]
tf_utilities = TensorflowUtilities()
for arg in arguments:
    tf_utilities.load_image_into_numpy_array(arg)

tf_utilities.run_predictor()

