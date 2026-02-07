import albumentations as A
from albumentations.pytorch import ToTensorV2

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tqdm.notebook import tqdm
import os

from yolox import cxcywh2xyxy

def get_inference_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def postprocess_predictions(outputs, conf_threshold=0.3, iou_threshold=0.45):
    # outputs: tensor of shape [1, num_anchors_all, 5 + num_classes]
    # where 5 is (cx, cy, w, h, obj_conf)

    # Assuming batch size is 1 for visualization for now
    output = outputs[0] # Get predictions for the first image in the batch

    # Filter by object confidence
    obj_conf = output[:, 4]
    scores = obj_conf # For single class, obj_conf is enough for score

    valid_preds_indices = torch.where(scores >= conf_threshold)[0]
    output = output[valid_preds_indices]
    scores = scores[valid_preds_indices]

    if output.shape[0] == 0:
        return torch.empty((0, 6)) # Return empty tensor if no valid predictions

    bboxes_cxcywh = output[:, :4] # (cx, cy, w, h)
    bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh) # Convert to xyxy for NMS

    # Apply NMS
    keep = nms(bboxes_xyxy, scores, iou_threshold)
    final_boxes = bboxes_xyxy[keep]
    final_scores = scores[keep]
    final_classes = output[keep, 5:].argmax(dim=1) # Get class with highest score

    # Combine into (x1, y1, x2, y2, score, class_id) format
    return torch.cat([final_boxes, final_scores.unsqueeze(1), final_classes.unsqueeze(1).float()], dim=1)

def nms(bboxes, scores, iou_threshold=0.45):
    # bboxes: (N, 4) in xyxy format
    # scores: (N,) confidence scores
    # iou_threshold: threshold for NMS

    if scores.numel() == 0: # Handle empty scores tensor
        return torch.tensor([], dtype=torch.long)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True) # Sort by confidence in descending order, avoiding [::-1]

    keep = []
    while order.numel() > 0: # Changed from order.size > 0 to order.numel() > 0
        idx = order[0]
        keep.append(idx)

        if order.numel() == 1:
            break

        xx1 = torch.max(x1[idx], x1[order[1:]])
        yy1 = torch.max(y1[idx], y1[order[1:]])
        xx2 = torch.min(x2[idx], x2[order[1:]])
        yy2 = torch.min(y2[idx], y2[order[1:]])

        w = torch.max(torch.tensor(0.0), xx2 - xx1)
        h = torch.max(torch.tensor(0.0), yy2 - yy1)

        inter = w * h
        ovr = inter / (areas[idx] + areas[order[1:]] - inter + 1e-6)

        inds = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1] 

    return torch.tensor(keep, dtype=torch.long)   


def visualize_detection_inference_streamlit(
    model,
    pil_image_input,
    device,
    conf_threshold=0.3,
    iou_threshold=0.45,
    patch_size=1024,
    overlap_ratio=0.2,
    image_name="Uploaded Image"
):
    all_bboxes_global_coord = []
    model.eval()

    # Convert PIL Image to OpenCV format (numpy array) for processing
    original_image = np.array(pil_image_input)
    # PIL Image is usually RGB, no need for cv2.cvtColor unless it's BGR
    # If pil_image_input was BGR, we would need: original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    h_orig, w_orig, _ = original_image.shape
    st.write(f"Размер изображения (HxW): {h_orig}x{w_orig}")

    inference_transforms = get_inference_transforms(img_size=640)
    model_input_size = 640

    with torch.no_grad():
        if w_orig > patch_size or h_orig > patch_size:
            st.write(f"Изображение большого размера ({w_orig}x{h_orig}), разбивка для обработки на части {patch_size}x{patch_size} с {overlap_ratio*100:.1f}% перектытия.")
            stride = int(patch_size * (1 - overlap_ratio))
            if stride <= 0:
                stride = 1 # Ensure stride is at least 1

            y_coords = list(range(0, h_orig - patch_size + 1, stride))
            if (h_orig - patch_size) % stride != 0:
                y_coords.append(h_orig - patch_size)
            if not y_coords: y_coords = [0]

            x_coords = list(range(0, w_orig - patch_size + 1, stride))
            if (w_orig - patch_size) % stride != 0:
                x_coords.append(w_orig - patch_size)
            if not x_coords: x_coords = [0]

            progress_bar = st.progress(0, text="Обработка изображения по частям...")
            total_patches = len(y_coords) * len(x_coords)
            processed_patches = 0

            for y in y_coords:
                for x in x_coords:
                    patch = original_image[y : y + patch_size, x : x + patch_size].copy()

                    transformed_patch = inference_transforms(image=patch)['image']
                    input_tensor = transformed_patch.unsqueeze(0).to(device)

                    predictions = model(input_tensor)

                    processed_preds = postprocess_predictions(predictions, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

                    if processed_preds.shape[0] > 0:
                        scale_factor_x = patch.shape[1] / model_input_size
                        scale_factor_y = patch.shape[0] / model_input_size

                        for pred_box in processed_preds:
                            x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id = pred_box.cpu().numpy()

                            global_x1 = x1_scaled * scale_factor_x + x
                            global_y1 = y1_scaled * scale_factor_y + y
                            global_x2 = x2_scaled * scale_factor_x + x
                            global_y2 = y2_scaled * scale_factor_y + y

                            all_bboxes_global_coord.append([global_x1, global_y1, global_x2, global_y2, score, class_id])

                    processed_patches += 1
                    progress_bar.progress(processed_patches / total_patches)
            progress_bar.empty()
        else:
            st.write("Изображение проходит по размеру, обработка изображения целиком")
            transformed_image = inference_transforms(image=original_image)['image']
            input_tensor = transformed_image.unsqueeze(0).to(device)

            predictions = model(input_tensor)

            processed_preds = postprocess_predictions(predictions, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

            if processed_preds.shape[0] > 0:
                scale_factor_x = w_orig / model_input_size
                scale_factor_y = h_orig / model_input_size

                for pred_box in processed_preds:
                    x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id = pred_box.cpu().numpy()
                    global_x1 = x1_scaled * scale_factor_x
                    global_y1 = y1_scaled * scale_factor_y
                    global_x2 = x2_scaled * scale_factor_x
                    global_y2 = y2_scaled * scale_factor_y
                    all_bboxes_global_coord.append([global_x1, global_y1, global_x2, global_y2, score, class_id])

    if len(all_bboxes_global_coord) > 0:
        all_bboxes_global_coord = np.array(all_bboxes_global_coord)
        boxes_to_nms = torch.tensor(all_bboxes_global_coord[:, :4], dtype=torch.float32)
        scores_to_nms = torch.tensor(all_bboxes_global_coord[:, 4], dtype=torch.float32)

        keep_indices = nms(boxes_to_nms, scores_to_nms, iou_threshold=iou_threshold)
        final_global_bboxes = all_bboxes_global_coord[keep_indices.cpu().numpy()]
    else:
        final_global_bboxes = np.empty((0, 6))

    display_image = original_image.copy()
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15)) # Create a new figure
    ax.imshow(display_image)
    ax.axis('off')

    for bbox_data in final_global_bboxes:
        x1, y1, x2, y2, score, class_id = bbox_data
        box_color = cmap(norm(score))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=box_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{score:.2f}', color=box_color, fontsize=8)

    # Создаем колонки, чтобы карточка не растягивалась на всю ширину
    d_col1, d_col2 = st.columns([1, 3]) 

    with d_col1:
        st.metric(
            label="Всего найдено объектов: ", 
            value=len(final_global_bboxes),
            help="Количество объектов, найденных детектором после фильтрации"
        )

    st.pyplot(fig) 
    plt.close(fig)

    return final_global_bboxes
