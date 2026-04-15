```

*Note: For NVIDIA Jetson, it is recommended to use `onnxruntime-gpu` for hardware acceleration.*

## Usage

The following example demonstrates how to initialize and use the `SlimRapidOCR` pipeline:

```python
from SlimRapidOCR import SlimRapidOCR

# Initialize the OCR pipeline with ONNX models
ocr = SlimRapidOCR(
    det_model="path/to/det.onnx",
    cls_model="path/to/cls.onnx",
    rec_model="path/to/rec.onnx",
    rec_dict="path/to/dict.txt"
)

# Process an image
results = ocr("test.png")

# Print results: [[bounding_box, text, confidence], ...]
for box, text, confidence in results:
    print(f"Text: {text} | Confidence: {confidence:.2f}")
```

## Credits

This project is built upon the amazing work of:

- [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) - The original RapidOCR project.
- [monkt (Hugging Face)](https://huggingface.co/monkt) - Provider of pre-trained ONNX models.

## License

This project follows the licensing of its upstream components. Refer to RapidOCR for details.
