# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
from io import BytesIO
from PIL import Image

model_inputs = {'prompt': 'realistic field of grass'}

res = requests.post('http://localhost:8000/', json = model_inputs)

image_byte_strings = res.json()["images_base64"]

for i,image_byte_string in enumerate(image_byte_strings):
  image_encoded = image_byte_string.encode('utf-8')
  image_bytes = BytesIO(base64.b64decode(image_encoded))
  image = Image.open(image_bytes)
  image.save("output{}.jpg".format(str(i))
