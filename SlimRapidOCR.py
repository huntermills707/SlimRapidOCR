import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon
import math
from typing import Tuple

# --- Utilities ---

def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Crops and rotates a region of an image specified by four points.
    
    Args:
        img: The source image.
        points: A 4x2 array of points defining the bounding box.
        
    Returns:
        The cropped and potentially rotated sub-image.
    """
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    )
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32(
        [[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]]
    )
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
    """
    Sorts bounding boxes from top to bottom and then left to right.
    
    Args:
        dt_boxes: Array of detected bounding boxes.
        
    Returns:
        Sorted bounding boxes.
    """
    if len(dt_boxes) == 0:
        return dt_boxes
    # Stable sort by y (top to bottom)
    y_coords = dt_boxes[:, 0, 1]
    y_order = np.argsort(y_coords, kind="stable")
    boxes_y_sorted = dt_boxes[y_order]
    y_sorted = y_coords[y_order]
    # Assign line IDs based on adjacent y differences
    dy = np.diff(y_sorted)
    line_increments = (dy >= 10).astype(np.int32)
    line_ids = np.concatenate([[0], np.cumsum(line_increments)])
    # Within each line group, sort by x (left to right)
    x_coords = boxes_y_sorted[:, 0, 0]
    final_order = np.lexsort((x_coords, line_ids))
    return boxes_y_sorted[final_order]

# --- Detection ---

class DBPostProcess:
    """
    Post-processing for DB (Differentiable Binarization) text detection.
    """
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.6):
        """
        Initializes DB post-processor.
        
        Args:
            thresh: Threshold for binarizing the probability map.
            box_thresh: Threshold for the average score of a box.
            max_candidates: Maximum number of candidate boxes.
            unclip_ratio: Ratio for expanding the detected box.
        """
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def __call__(self, pred: np.ndarray, ori_shape: Tuple[int, int]):
        """
        Applies post-processing to the detector's output.
        
        Args:
            pred: Prediction map from the detector.
            ori_shape: Original image shape (height, width).
            
        Returns:
            A tuple containing (boxes, scores).
        """
        src_h, src_w = ori_shape
        pred = pred[0, 0, :, :]
        segmentation = pred > self.thresh
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, src_w, src_h)
        return boxes, scores

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        Extracts bounding boxes from a binarized bitmap.
        
        Args:
            pred: Probability map.
            bitmap: Binarized map.
            dest_width: Target image width.
            dest_height: Target image height.
            
        Returns:
            A tuple containing (boxes, scores).
        """
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score: continue
            box = self.unclip(points).reshape(-1, 2)
            box, sside = self.get_mini_boxes(box.reshape(-1, 1, 2))
            if sside < self.min_size + 2: continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    def get_mini_boxes(self, contour):
        """
        Finds the minimum area bounding box for a contour and normalizes the points.
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = (0, 1, 2, 3) if points[1][1] <= points[0][1] else (1, 0, 2, 3)
        if points[1][1] > points[0][1]: index_1, index_4 = 0, 1
        else: index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]: index_2, index_3 = 2, 3
        else: index_2, index_3 = 3, 2
        box = np.array([points[index_1], points[index_2], points[index_3], points[index_4]])
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        Calculates the average score of a probability map within a bounding box.
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin, xmax = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1), np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin, ymax = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1), np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [(box - [xmin, ymin]).astype(np.int32)], 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def unclip(self, box):
        """
        Expands a bounding box by the unclip ratio.
        """
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance))

class TextDetector:
    """
    Text detector using an ONNX model.
    """
    def __init__(self, model_path):
        """
        Initializes the text detector.
        
        Args:
            model_path: Path to the ONNX detection model.
        """
        self.session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider'])
        self.postprocess_op = DBPostProcess()

    def preprocess(self, img):
        """
        Preprocesses an image for the detection model.
        
        Args:
            img: Input image.
            
        Returns:
            Preprocessed image as a numpy array.
        """
        h, w = img.shape[:2]
        ratio = 960 / max(h, w)
        if ratio > 1: ratio = 1.0
        rh, rw = int(round(h * ratio / 32) * 32), int(round(w * ratio / 32) * 32)
        img = cv2.resize(img, (rw, rh))
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img = img.transpose((2, 0, 1))[np.newaxis, :]
        return img

    def __call__(self, img):
        """
        Performs text detection on an image.
        
        Args:
            img: Input image.
            
        Returns:
            A tuple containing (detected_boxes, scores).
        """
        pre_img = self.preprocess(img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: pre_img})
        boxes, scores = self.postprocess_op(outputs[0], img.shape[:2])
        if len(boxes) > 0:
            boxes = sorted_boxes(boxes)
        return boxes, scores

# --- Classifier ---

class TextClassifier:
    """
    Text orientation classifier using an ONNX model.
    """
    def __init__(self, model_path):
        """
        Initializes the text classifier.
        
        Args:
            model_path: Path to the ONNX classification model.
        """
        self.session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider'])
        self.labels = ['0', '90', '180', '270']

    def resize_norm_img(self, img):
        """
        Resizes and normalizes an image for the classification model.
        """
        img_c, img_h, img_w = 3, 224, 224
        resized_image = cv2.resize(img, (img_w, img_h))
        resized_image = (resized_image.astype(np.float32).transpose((2, 0, 1)) / 255.0 - 0.5) / 0.5
        return resized_image

    def __call__(self, img_list):
        """
        Classifies and corrects the orientation of a list of images.
        
        Args:
            img_list: List of images or a single image.
            
        Returns:
            List of tuples: (rotated_image, label, score).
        """
        if not isinstance(img_list, list): img_list = [img_list]
        res = []
        for img in img_list:
            norm_img = self.resize_norm_img(img)[np.newaxis, :]
            prob = self.session.run(None, {self.session.get_inputs()[0].name: norm_img})[0]
            idx = prob.argmax()
            label, score = self.labels[idx], prob[0, idx]
            if label == '180':
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif label == '90':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif label == '270':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            res.append((img, label, score))
        return res

# --- Recognition ---

class TextRecognizer:
    """
    Text recognizer using an ONNX model.
    """
    def __init__(self, model_path, dict_path=None):
        """
        Initializes the text recognizer.
        
        Args:
            model_path: Path to the ONNX recognition model.
            dict_path: Optional path to the character dictionary file.
        """
        self.session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider'])
        
        if dict_path:
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.character = ['blank'] + [line.strip('\r\n') for line in f.readlines()] + [' ']
        else:
            meta = self.session.get_modelmeta().custom_metadata_map
            if 'character' in meta:
                self.character = ['blank'] + meta['character'].splitlines() + [' ']
            else:
                raise ValueError("No 'character' metadata found in model and no dict_path provided.")

    def resize_norm_img(self, img, max_wh_ratio):
        """
        Resizes and normalizes an image for the recognition model, with padding.
        """
        img_c, img_h, img_w = 3, 48, int(48 * max_wh_ratio)
        h, w = img.shape[:2]
        rw = min(img_w, int(math.ceil(img_h * (w / h))))
        resized_image = cv2.resize(img, (rw, img_h))
        resized_image = (resized_image.astype(np.float32).transpose((2, 0, 1)) / 255.0 - 0.5) / 0.5
        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, :rw] = resized_image
        return padding_im

    def __call__(self, img_list):
        """
        Recognizes text in a list of images.
        
        Args:
            img_list: List of images or a single image.
            
        Returns:
            List of tuples: (recognized_text, confidence_score).
        """
        if not isinstance(img_list, list): img_list = [img_list]
        max_wh_ratio = max([img.shape[1]/img.shape[0] for img in img_list] + [320/48])
        results = []
        for img in img_list:
            norm_img = self.resize_norm_img(img, max_wh_ratio)[np.newaxis, :]
            preds = self.session.run(None, {self.session.get_inputs()[0].name: norm_img})[0]
            # CTC Decode
            idx = preds.argmax(axis=2)[0]
            prob = preds.max(axis=2)[0]
            char_list, conf_list = [], []
            for i in range(len(idx)):
                if idx[i] != 0 and (not (i > 0 and idx[i] == idx[i-1])):
                    char_list.append(self.character[idx[i]])
                    conf_list.append(prob[i])
            results.append(("".join(char_list), np.mean(conf_list) if conf_list else 0.0))
        return results

# --- Main RapidOCR_x Slim ---

class SlimRapidOCR:
    """
    Combined OCR pipeline: Detection -> Classification -> Recognition.
    """
    def __init__(self, det_model, cls_model, rec_model, rec_dict=None):
        """
        Initializes the OCR pipeline.
        
        Args:
            det_model: Path to the detection model.
            cls_model: Path to the classification model.
            rec_model: Path to the recognition model.
            rec_dict: Optional path to the recognition dictionary.
        """
        self.text_detector = TextDetector(det_model)
        self.text_classifier = TextClassifier(cls_model)
        self.text_recognizer = TextRecognizer(rec_model, rec_dict)

    def __call__(self, img_path):
        """
        Processes an image and returns full OCR results.
        
        Args:
            img_path: Path to the input image.
            
        Returns:
            A list of results: [[bounding_box, text, confidence], ...].
        """
        img = cv2.imread(img_path)
        if img is None: return None
        
        # 1. Detection
        dt_boxes, _ = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) == 0: return None
        
        # 2. Crop & 3. Cls
        img_crop_list = [get_rotate_crop_image(img, box) for box in dt_boxes]
        cls_res = self.text_classifier(img_crop_list)
        cls_imgs = [r[0] for r in cls_res]
        
        # 4. Recognition
        rec_res = self.text_recognizer(cls_imgs)
        
        # Combine results
        return [[dt_boxes[i].tolist(), rec_res[i][0], rec_res[i][1]] for i in range(len(dt_boxes))]

if __name__ == "__main__":
    # Example usage
    ocr = SlimRapidOCR(
        det_model="monkt-paddle/paddleocr-onnx/detection/v5/det.onnx",
        cls_model="monkt-paddle/paddleocr-onnx/preprocessing/doc-orientation/PP-LCNet_x1_0_doc_ori.onnx",
        rec_model="monkt-paddle/paddleocr-onnx/languages/english/rec.onnx",
        rec_dict="monkt-paddle/paddleocr-onnx/languages/english/dict.txt"
    )

    print(ocr("test.png"))
