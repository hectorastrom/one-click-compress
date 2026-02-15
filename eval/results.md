# YOLO
YOLO completely fails from quanitization.

The best hypothesis is that the obscure output shape of YOLOv8 `[1,84,8000]`
which is totally imbalanced over classes. Dim 1 is structured as 0-3 bounding
boxes, and 4-83 as class predictions. Since the bounding boxes have much higher
values than the class predictions (they represent absolute coordinates), the
attempted preservation of the output range completely skews the data
post-quantization. 

As such, we pivoted to using Resnet as a demonstrative model.

# Resnet
Argmax Accuracy Benchmark (categorical models only)
------------------------------------------------------------
FP32 model:      weights/resnet50.pt2 (pt2)
Quantized model: out/resnet50_int8_xnnpack.pte (pte)
Dataset:         data/imagenette_calibration.pt
Batch size:      1
------------------------------------------------------------
Samples evaluated:      500
FP32 top-1 accuracy:    0.9640
INT8 top-1 accuracy:    0.9700
Prediction agreement:   0.9900
Accuracy delta (INT8):  +0.0060