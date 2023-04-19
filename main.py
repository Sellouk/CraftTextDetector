# import Craft class
from craft_text_detector import Craft

# set image path and export folder directory
image = 'input directory' # can be filepath, PIL image or numpy array
output_dir = 'output directory'

# create a craft instance
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# apply craft text detection and export detected regions to output directory
prediction_result = craft.detect_text(image)

# unload models from ram/gpu
craft.unload_craftnet_model()
craft.unload_refinenet_model()