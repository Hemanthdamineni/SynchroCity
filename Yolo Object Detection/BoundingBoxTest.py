import cv2
import os
import sys

def visualize_bbox_debug(image_path, label_path):
    """Debug visualization showing both coordinate systems"""
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"\nImage dimensions: {w}x{h}")
    
    # Create two copies for comparison
    img_normal = img.copy()
    img_flipped = img.copy()
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                print(f"\nYOLO coordinates: cx={cx:.3f}, cy={cy:.3f}, w={bw:.3f}, h={bh:.3f}")
                
                # Convert to pixel coordinates (assuming YOLO format)
                center_x_px = cx * w
                center_y_px = cy * h
                width_px = bw * w
                height_px = bh * h
                
                # Normal interpretation (Y from top)
                x1_normal = int(center_x_px - width_px / 2)
                y1_normal = int(center_y_px - height_px / 2)
                x2_normal = int(center_x_px + width_px / 2)
                y2_normal = int(center_y_px + height_px / 2)
                
                # Flipped interpretation (Y from bottom)
                y1_flipped = int(h - (center_y_px + height_px / 2))
                y2_flipped = int(h - (center_y_px - height_px / 2))
                x1_flipped = x1_normal
                x2_flipped = x2_normal
                
                print(f"Normal (Y from top): ({x1_normal}, {y1_normal}) to ({x2_normal}, {y2_normal})")
                print(f"Flipped (Y from bottom): ({x1_flipped}, {y1_flipped}) to ({x2_flipped}, {y2_flipped})")
                
                # Draw on both versions
                cv2.rectangle(img_normal, (x1_normal, y1_normal), (x2_normal, y2_normal), (0, 255, 0), 2)
                cv2.putText(img_normal, "Normal Y", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.rectangle(img_flipped, (x1_flipped, y1_flipped), (x2_flipped, y2_flipped), (0, 0, 255), 2)
                cv2.putText(img_flipped, "Flipped Y", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Stack images side by side
    comparison = cv2.hconcat([img_normal, img_flipped])
    
    # Save comparison
    output_path = image_path.replace('.png', '_comparison.png')
    cv2.imwrite(output_path, comparison)
    print(f"\nComparison saved to: {output_path}")
    
    # Display if possible
    try:
        cv2.imshow("Left: Normal Y (top=0) | Right: Flipped Y (bottom=0)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Cannot display (no GUI available)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bbox_diagnostic.py <path_to_image>")
        print("\nOr run without args to test latest image in YOLODataset/visualizations/")
        
        # Find latest visualization
        viz_dir = "YOLODataset/visualizations"
        if os.path.exists(viz_dir):
            files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
            if files:
                latest = max(files, key=lambda f: os.path.getctime(os.path.join(viz_dir, f)))
                img_path = os.path.join(viz_dir, latest)
                
                # Find corresponding label
                base_name = latest.replace('.png', '')
                for split in ['train', 'val']:
                    label_path = f"YOLODataset/labels/{split}/{base_name}.txt"
                    if os.path.exists(label_path):
                        print(f"Testing: {img_path}")
                        visualize_bbox_debug(img_path, label_path)
                        break
        sys.exit(0)
    
    img_path = sys.argv[1]
    
    # Try to find corresponding label file
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    label_path = None
    for split in ['train', 'val']:
        candidate = f"YOLODataset/labels/{split}/{base_name}.txt"
        if os.path.exists(candidate):
            label_path = candidate
            break
    
    if label_path is None:
        print(f"Could not find label file for {img_path}")
        sys.exit(1)
    
    visualize_bbox_debug(img_path, label_path)