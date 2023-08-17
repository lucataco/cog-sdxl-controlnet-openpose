# thibaud/controlnet-openpose-sdxl-1.0 Cog model

This is an implementation of the [thibaud/controlnet-openpose-sdxl-1.0](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@demo.jpg -i prompt="a latina ballerina, romantic sunset, 4k photo"

## Example:

Inputs:

"a latina ballerina, romantic sunset, 4k photo"

![alt text](demo.jpg)

Output

![alt text](output.png)
