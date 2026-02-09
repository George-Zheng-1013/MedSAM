import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ShapeTemplate:
    area_ratio: float
    circularity: float
    elongation: float


@dataclass
class TopologyPostprocessConfig:
    min_hole_area: int = 32
    hole_overlap_allow_ratio: float = 0.2
    myo_cavity_min_ratio: float = 0.08
    template_match_threshold: float = 0.55
    diffusion_iters: int = 12
    diffusion_edge_weight: float = 0.6
    diffusion_enabled: bool = True
    crf_enabled: bool = True
    crf_iters: int = 10
    class_priority: Tuple[int, ...] = (1, 2, 3)
    templates: Optional[Dict[int, List[ShapeTemplate]]] = None


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    if gray.max() <= 1.0:
        gray = (gray * 255.0).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)
    return gray


def _find_holes(binary_mask: np.ndarray) -> np.ndarray:
    inv = (binary_mask == 0).astype(np.uint8)
    flood = inv.copy()
    h, w = inv.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    holes = (flood == 1).astype(np.uint8)
    return holes


def _hole_components(holes: np.ndarray) -> List[Dict[str, np.ndarray]]:
    num_labels, labels = cv2.connectedComponents(holes, connectivity=8)
    comps = []
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id).astype(np.uint8)
        area = int(comp_mask.sum())
        if area == 0:
            continue
        ys, xs = np.where(comp_mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        centroid = (float(xs.mean()), float(ys.mean()))
        contour, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contour:
            perimeter = float(cv2.arcLength(contour[0], True))
        else:
            perimeter = 0.0
        circularity = 0.0
        if perimeter > 0.0:
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        elongation = 1.0
        if contour and len(contour[0]) >= 5:
            (cx, cy), (ma, mi), _ = cv2.fitEllipse(contour[0])
            if mi > 0:
                elongation = float(ma / mi)
        comps.append(
            {
                "mask": comp_mask,
                "area": area,
                "centroid": centroid,
                "bbox": (x_min, y_min, x_max, y_max),
                "circularity": circularity,
                "elongation": elongation,
            }
        )
    return comps


def _template_score(desc: Dict[str, float], templ: ShapeTemplate) -> float:
    diffs = []
    for key, target in (
        ("area_ratio", templ.area_ratio),
        ("circularity", templ.circularity),
        ("elongation", templ.elongation),
    ):
        value = float(desc.get(key, 0.0))
        denom = max(float(target), 1e-6)
        diffs.append(min(abs(value - target) / denom, 1.0))
    return 1.0 - float(np.mean(diffs))


def _default_templates() -> Dict[int, List[ShapeTemplate]]:
    return {
        2: [
            ShapeTemplate(area_ratio=0.2, circularity=0.55, elongation=1.3),
            ShapeTemplate(area_ratio=0.12, circularity=0.5, elongation=1.6),
        ]
    }


def _should_allow_hole(
    hole: Dict[str, np.ndarray],
    class_id: int,
    class_area: int,
    multi_mask: np.ndarray,
    config: TopologyPostprocessConfig,
) -> bool:
    if class_area <= 0:
        return False
    area_ratio = float(hole["area"]) / float(class_area)
    desc = {
        "area_ratio": area_ratio,
        "circularity": float(hole["circularity"]),
        "elongation": float(hole["elongation"]),
    }

    overlap = multi_mask[hole["mask"] > 0]
    if overlap.size > 0:
        non_bg_ratio = float(np.sum(overlap > 0)) / float(overlap.size)
        if non_bg_ratio >= config.hole_overlap_allow_ratio:
            return True

    templates = (
        config.templates if config.templates is not None else _default_templates()
    )
    for templ in templates.get(class_id, []):
        if _template_score(desc, templ) >= config.template_match_threshold:
            return True

    if class_id == 2 and area_ratio >= config.myo_cavity_min_ratio:
        if desc["circularity"] >= 0.4:
            return True

    return False


def _diffusion_fill(
    image: np.ndarray,
    class_mask: np.ndarray,
    hole_mask: np.ndarray,
    iters: int,
    edge_weight: float,
) -> np.ndarray:
    gray = _to_uint8_gray(image)
    edge = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    prob = class_mask.astype(np.float32)
    hole_idx = hole_mask > 0
    if not np.any(hole_idx):
        return class_mask

    for _ in range(max(iters, 1)):
        smooth = cv2.blur(prob, (3, 3))
        weight = 1.0 - (edge_weight * edge)
        prob[hole_idx] = (
            weight[hole_idx] * smooth[hole_idx]
            + (1.0 - weight[hole_idx]) * prob[hole_idx]
        )

    filled = class_mask.copy()
    filled[hole_idx] = (prob[hole_idx] > 0.5).astype(np.uint8)
    return filled


def _close_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _build_soft_probs(mask: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = mask.shape
    probs = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        binary = (mask == c).astype(np.float32)
        probs[c] = cv2.GaussianBlur(binary, (5, 5), 0)

    sum_probs = probs.sum(axis=0, keepdims=True)
    if np.all(sum_probs == 0):
        for c in range(num_classes):
            probs[c] = (mask == c).astype(np.float32)
        sum_probs = probs.sum(axis=0, keepdims=True)

    probs = probs / np.clip(sum_probs, 1e-6, None)
    return probs


def _apply_crf(
    image: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    iters: int,
    soft_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=2)
    else:
        image_rgb = image

    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255.0).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)

    h, w = mask.shape

    probs = soft_probs.astype(np.float32)
    if probs.ndim == 2:
        # Binary foreground prob -> two-class softmax-like probs
        fg = np.clip(probs, 0.0, 1.0)
        bg = 1.0 - fg
        probs = np.stack([bg, fg], axis=0)
    elif probs.ndim == 3 and probs.shape[-1] == num_classes:
        probs = np.transpose(probs, (2, 0, 1))
    if probs.shape[0] != num_classes or probs.shape[1:] != (h, w):
        raise ValueError("soft_probs must be (C,H,W) or (H,W,C) or (H,W)")
    
    eps = 1e-4
    probs = probs * (1.0 - eps) + (eps / float(num_classes))
    unary = unary_from_softmax(probs)

    d = dcrf.DenseCRF2D(w, h, num_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=5, compat=5)
    d.addPairwiseBilateral(sxy=50, srgb=10, rgbim=image_rgb, compat=15)

    q = d.inference(max(iters, 1))
    refined = np.array(q).reshape((num_classes, h, w)).argmax(axis=0).astype(np.uint8)
    return refined


def apply_topology_postprocess(
    mask: np.ndarray,
    image: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    soft_probs: Optional[np.ndarray] = None,
    config: Optional[TopologyPostprocessConfig] = None,
) -> Dict[str, object]:
    if config is None:
        config = TopologyPostprocessConfig()

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    refined = mask.copy().astype(np.uint8)
    h, w = refined.shape

    stats = {
        "holes_total": 0,
        "holes_filled": 0,
        "holes_allowed": 0,
    }

    for class_id in config.class_priority:
        class_mask = (refined == class_id).astype(np.uint8)
        class_mask = _close_mask(class_mask)
        allowed = (refined == 0) | (refined == class_id)
        class_mask = (class_mask & allowed).astype(np.uint8)
        class_area = int(class_mask.sum())
        if class_area == 0:
            continue

        holes = _find_holes(class_mask)
        components = _hole_components(holes)

        for hole in components:
            if hole["area"] < config.min_hole_area:
                continue

            stats["holes_total"] += 1
            allow = _should_allow_hole(hole, class_id, class_area, refined, config)
            if allow:
                stats["holes_allowed"] += 1
                continue

            fill_mask = hole["mask"].astype(bool) & (refined == 0)
            if not np.any(fill_mask):
                continue

            if config.diffusion_enabled:
                class_mask = _diffusion_fill(
                    image,
                    class_mask,
                    fill_mask.astype(np.uint8),
                    config.diffusion_iters,
                    config.diffusion_edge_weight,
                )
            else:
                class_mask[fill_mask] = 1

            refined[fill_mask] = class_id
            stats["holes_filled"] += 1

        refined[refined == class_id] = 0
        refined[class_mask > 0] = class_id

    if config.crf_enabled:
        num_classes = int(refined.max()) + 1
        refined = _apply_crf(
            image, refined, num_classes, config.crf_iters, soft_probs=soft_probs
        )

    return {"mask": refined, "stats": stats}
