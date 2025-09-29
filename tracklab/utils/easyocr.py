from typing import Union

import numpy as np
import torch


def bbox_easyocr_to_image_ltwh(
    easy_ocr_bbox: Union[list, np.ndarray],
    bbox_tlwh: torch.Tensor,
) -> torch.Tensor:
    """Convert EasyOCR bbox to image coordinates in LTWH format.

    Args:
        easy_ocr_bbox: EasyOCR bounding box coordinates.
        bbox_tlwh: Original bounding box in LTWH format.

    Returns:
        Converted bounding box in image coordinates.
    """
    easy_ocr_bbox = np.array(easy_ocr_bbox)  # only need the top left point
    tl = easy_ocr_bbox[0]
    tr = easy_ocr_bbox[1]
    br = easy_ocr_bbox[2]
    bl = easy_ocr_bbox[3]

    width = tr[1] - tl[1]
    height = bl[0] - tl[0]

    jn_bbox = bbox_tlwh.clone()
    jn_bbox[0] = jn_bbox[0] + tl[0]
    jn_bbox[1] = jn_bbox[1] + tl[1]
    jn_bbox[2] = width
    jn_bbox[3] = height
    return jn_bbox
