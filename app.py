#!/usr/bin/env python3
"""
YOLOv12-DINO Gradio Web Interface

A web-based interface for YOLOv12-DINO object detection using Gradio.
Upload images and get real-time object detection results with adjustable parameters.
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
from typing import Tuple, Optional
import traceback

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from inference import YOLOInference
    from ultralytics.utils import LOGGER
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)

# Global variables
model_instance = None
default_weights = "runs/detect/train/weights/best.pt"

def load_model(weights_path: str, device: str = "cpu") -> bool:
    """Load the YOLO model with specified weights."""
    global model_instance
    try:
        if not Path(weights_path).exists():
            return False
        
        # Clear any existing model instance
        model_instance = None
        
        # Force reload modules to ensure latest code
        import importlib
        import sys
        if 'inference' in sys.modules:
            importlib.reload(sys.modules['inference'])
        
        model_instance = YOLOInference(
            weights=weights_path,
            conf=0.25,
            iou=0.7,
            imgsz=640,
            device=device,
            verbose=True  # Enable verbose for debugging
        )
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_image(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    image_size: int
) -> Tuple[Optional[np.ndarray], str]:
    """
    Perform object detection on the input image.
    
    Args:
        image: Input image as numpy array
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
        image_size: Input image size for model
        
    Returns:
        Tuple of (annotated_image, results_text)
    """
    global model_instance
    
    if model_instance is None:
        return None, "❌ Model not loaded. Please load a model first."
    
    if image is None:
        return None, "❌ No image provided."
    
    try:
        # Update model parameters
        model_instance.conf = conf_threshold
        model_instance.iou = iou_threshold
        model_instance.imgsz = image_size
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_file.name, image_bgr)
            tmp_path = tmp_file.name
        
        try:
            # Run inference
            results = model_instance.predict_single(
                source=tmp_path,
                save=False,
                show=False
            )
            
            if not results:
                return image, "❌ No results returned from model."
            
            result = results[0]
            
            # Get annotated image
            annotated_img = result.plot()
            
            # Convert BGR back to RGB for display
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Generate results text
            results_text = generate_results_text(result)
            
            return annotated_img, results_text
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        error_msg = f"❌ Error during inference: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg

def generate_results_text(result) -> str:
    """Generate formatted results text from detection results."""
    if result.boxes is None or len(result.boxes) == 0:
        return "🔍 No objects detected in the image."
    
    detections = result.boxes
    num_detections = len(detections)
    
    results_text = f"✅ **Detected {num_detections} object(s):**\n\n"
    
    # Get class names
    if hasattr(result, 'names') and result.names:
        class_names = result.names
    else:
        class_names = {i: f"Class_{i}" for i in range(100)}  # Fallback
    
    # Count detections by class
    class_counts = {}
    detection_details = []
    
    for i, (box, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
        cls_id = int(cls)
        cls_name = class_names.get(cls_id, f"Class_{cls_id}")
        confidence = float(conf)
        
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        x1, y1, x2, y2 = box.tolist()
        detection_details.append(
            f"**{i+1}.** {cls_name} (confidence: {confidence:.2f})\n"
            f"   📍 Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})"
        )
    
    # Add class summary
    results_text += "📊 **Summary by class:**\n"
    for cls_name, count in sorted(class_counts.items()):
        results_text += f"• {cls_name}: {count}\n"
    
    results_text += "\n📋 **Detailed detections:**\n"
    results_text += "\n".join(detection_details)
    
    return results_text

def load_model_interface(weights_file, device):
    """Interface function for loading model through Gradio."""
    if weights_file is None:
        return "❌ Please select a weights file.", False
    
    try:
        # Save uploaded file temporarily
        temp_path = weights_file.name
        success = load_model(temp_path, device)
        
        if success:
            return f"✅ Model loaded successfully from {Path(temp_path).name}", True
        else:
            return "❌ Failed to load model. Check the weights file.", False
            
    except Exception as e:
        return f"❌ Error loading model: {str(e)}", False

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-image {
        max-height: 600px;
    }
    .results-text {
        max-height: 400px;
        overflow-y: auto;
    }
    """
    
    with gr.Blocks(css=css, title="YOLOv12-DINO Object Detection") as interface:
        gr.Markdown(
            """
            # 🎯 YOLOv12-DINO Object Detection
            
            Upload an image and adjust the detection parameters to perform object detection using YOLOv12-DINO.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 Model Loading")
                
                weights_file = gr.File(
                    label="Upload Model Weights (.pt file)",
                    file_types=[".pt"],
                    type="filepath"
                )
                
                device_choice = gr.Radio(
                    choices=["cpu", "cuda", "mps"],
                    value="cpu",
                    label="Device",
                    info="Select computation device"
                )
                
                load_btn = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                
                gr.Markdown("## ⚙️ Detection Parameters")
                
                conf_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.25,
                    step=0.01,
                    label="Confidence Threshold",
                    info="Minimum confidence for detection"
                )
                
                iou_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.7,
                    step=0.01,
                    label="IoU Threshold",
                    info="IoU threshold for Non-Maximum Suppression"
                )
                
                size_slider = gr.Slider(
                    minimum=320,
                    maximum=1280,
                    value=640,
                    step=32,
                    label="Image Size",
                    info="Input image size for model"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 🖼️ Image Detection")
                
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                
                detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
                
                with gr.Row():
                    output_image = gr.Image(
                        label="Detection Results",
                        type="numpy",
                        height=400
                    )
                
                results_text = gr.Textbox(
                    label="Detection Details",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    elem_classes=["results-text"]
                )
        
        # Event handlers
        load_btn.click(
            fn=load_model_interface,
            inputs=[weights_file, device_choice],
            outputs=[model_status]
        )
        
        detect_btn.click(
            fn=predict_image,
            inputs=[input_image, conf_slider, iou_slider, size_slider],
            outputs=[output_image, results_text]
        )
        
        # Auto-detect on parameter change
        for component in [conf_slider, iou_slider, size_slider]:
            component.change(
                fn=predict_image,
                inputs=[input_image, conf_slider, iou_slider, size_slider],
                outputs=[output_image, results_text]
            )
        
        gr.Markdown(
            """
            ## 📋 Usage Instructions
            
            1. **Load Model**: Upload your trained YOLOv12-DINO weights file (.pt) and select device
            2. **Upload Image**: Select an image file for object detection
            3. **Adjust Parameters**: Fine-tune confidence and IoU thresholds as needed
            4. **Detect**: Click "Detect Objects" or parameters will auto-update results
            
            ## 🎛️ Parameter Guide
            
            - **Confidence Threshold**: Lower values detect more objects but may include false positives
            - **IoU Threshold**: Higher values remove more overlapping detections
            - **Image Size**: Larger sizes may improve accuracy but slow inference
            """
        )
    
    return interface

def main():
    """Main function to launch the Gradio interface."""
    # Try to auto-load default model if it exists
    if Path(default_weights).exists():
        print(f"Loading default model: {default_weights}")
        success = load_model(default_weights)
        if success:
            print("✅ Default model loaded successfully")
        else:
            print("❌ Failed to load default model")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with configuration
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()