import sys
from TensorflowUtilities import TensorflowUtilities

arguments = sys.argv
arguments = arguments[1:]
tf_utilities = TensorflowUtilities()
for arg in arguments:
    tf_utilities.load_image_into_numpy_array(arg)

tf_utilities.run_predictor()

