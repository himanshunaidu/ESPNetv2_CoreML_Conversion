# ESPNetv2 CoreML Conversion

This directory contains the minimal code to convert ESPNetv2 to CoreML format. The conversion is done using the `coremltools` library, which provides a convenient way to convert PyTorch models to CoreML.

## Environment Setup

To set up the environment, you can use the provided `espnetv2_coreml_conversion.yml` file. This file contains all the necessary dependencies to run the conversion script.

## Command

To convert the ESPNetv2 model to CoreML format, run the following command:

```bash
python coreml_conversion.py \
    --weight-path ./weights/espnetv2_s_2.0_city_1024x512.pth \
    --im-size 1024 512 \
    --s 2.0 \
    --outpath model_zoo/ \
    --img-path ./data/test.jpg \
    --dataset city
```