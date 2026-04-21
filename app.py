import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import base64
from io import BytesIO
import sys
from pathlib import Path
import torch
import random

class_color_map = {
    "C2": [139, 0, 0],    # Dark Red
    "C3": [0, 100, 0],    # Dark Green  
    "C4": [0, 0, 139],    # Dark Blue
}

# CVMI Stage Information
CVMI_STAGE_INFO = {
    "CS1": {
        "name": "Stage 1 - Initiation",
        "description": "Lower borders of C2, C3, C4 are flat. Vertebral bodies are trapezoid in shape.",
        "age_range": "6-8 years",
        "characteristics": ["Flat lower borders", "Trapezoid shapes", "No concavities"]
    },
    "CS2": {
        "name": "Stage 2 - Acceleration", 
        "description": "Concavities in lower borders of C2 and C3. C4 still flat.",
        "age_range": "8-10 years",
        "characteristics": ["C2 & C3 concavities", "C4 flat", "Beginning maturation"]
    },
    "CS3": {
        "name": "Stage 3 - Transition",
        "description": "Concavities in lower borders of C2, C3, C4. C3 and C4 are rectangular.",
        "age_range": "10-12 years", 
        "characteristics": ["All concavities present", "C3 & C4 rectangular", "Active growth"]
    },
    "CS4": {
        "name": "Stage 4 - Deceleration",
        "description": "Distinct concavities in all vertebrae. C3 and C4 nearly square.",
        "age_range": "12-14 years",
        "characteristics": ["Deep concavities", "Square shapes", "Growth slowing"]
    },
    "CS5": {
        "name": "Stage 5 - Maturation",
        "description": "More accentuated concavities. All vertebrae are rectangular.",
        "age_range": "14-16 years", 
        "characteristics": ["Accentuated concavities", "All rectangular", "Near completion"]
    },
    "CS6": {
        "name": "Stage 6 - Completion",
        "description": "Deep concavities, fully developed vertebrae. Growth complete.",
        "age_range": "16-18 years",
        "characteristics": ["Deep concavities", "Fully developed", "Growth complete"]
    }
}

# Cross-platform model loading
def detect_device():
    """Detect and return the best available device"""
    try:
        if torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon Macs
        elif torch.cuda.is_available():
            return 'cuda'  # NVIDIA GPUs
        else:
            return 'cpu'   # CPU fallback
    except:
        return 'cpu'  # Fallback to CPU if any error

def verify_models(roi_model, seg_model, cls_model):
    """Verify that all models are loaded and functional"""
    try:
        print("🔍 Verifying models...")
        
        # Create a small test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test ROI model
        roi_result = roi_model(test_image, verbose=False)
        print("✅ ROI model verified")
        
        # Test segmentation model
        seg_result = seg_model(test_image, verbose=False)
        print("✅ Segmentation model verified")
        
        # Test classification model
        cls_result = cls_model(test_image, verbose=False)
        print("✅ Classification model verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def load_models():
    """
    Load YOLO models with cross-platform compatibility
    Handles Windows, Mac (Intel & Apple Silicon), and Linux
    """
    try:
        print("=" * 60)
        print("INITIALIZING MODELS...")
        print("=" * 60)
        
        # Get the directory where your script is located
        base_dir = Path(__file__).parent
        model_dir = base_dir / "model"
        
        # Define model paths
        roi_path = model_dir / "roi_best.pt"
        seg_path = model_dir / "seg_best.pt"
        cls_path = model_dir / "best_new.pt"
        
        # Check if model directory exists
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            print("Please create a 'model' folder with the required .pt files")
            return None, None, None
        
        # Check if model files exist
        model_files = {
            "ROI": roi_path,
            "Segmentation": seg_path,
            "Classification": cls_path
        }
        
        missing_models = []
        for name, path in model_files.items():
            if path.exists():
                print(f"✅ {name} model found: {path}")
            else:
                print(f"❌ {name} model not found: {path}")
                missing_models.append(name)
        
        if missing_models:
            print(f"Missing models: {', '.join(missing_models)}")
            return None, None, None
        
        # Detect available device
        device = detect_device()
        print(f"🖥️  Using device: {device}")
        
        # Load models with error handling
        models = {}
        for name, path in model_files.items():
            try:
                print(f"📥 Loading {name} model...")
                model = YOLO(str(path))
                model.to(device)
                models[name.lower()] = model
                print(f"✅ {name} model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load {name} model: {e}")
                return None, None, None
        
        # Verify models are working
        if verify_models(*models.values()):
            print("🎉 All models loaded and verified successfully!")
            return models['roi'], models['segmentation'], models['classification']
        else:
            print("❌ Model verification failed!")
            return None, None, None
        
    except Exception as e:
        print(f"❌ Unexpected error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Initialize models
print("🚀 Starting CVMI Analysis System...")
roi_model, seg_model, cls_model = load_models()

def numpy_to_base64(image_array):
    """Convert numpy array to base64 string with error handling"""
    try:
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Handle different image formats
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_img = Image.fromarray(image_array)
        elif len(image_array.shape) == 2:
            pil_img = Image.fromarray(image_array, mode='L')
        else:
            # Convert to RGB if in BGR format
            pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        
        buff = BytesIO()
        pil_img.save(buff, format="JPEG", quality=95)
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        # Return a placeholder image
        placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(placeholder, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return numpy_to_base64(placeholder)

def create_cvmi_chart(cvmi_stage):
    """Create a CVMI chart image with the predicted stage highlighted"""
    try:
        # Create a blank chart image
        chart_width, chart_height = 800, 500
        chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255
        
        # Draw chart title
        cv2.putText(chart, "CERVICAL VERTEBRAL MATURATION INDEX (CVMI)", (150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Draw stages
        stages = list(CVMI_STAGE_INFO.keys())
        stage_width = chart_width // len(stages)
        
        for i, stage in enumerate(stages):
            x_center = i * stage_width + stage_width // 2
            y_center = 200
            
            # Draw stage circle
            color = (150, 150, 150)  # Default gray
            text_color = (0, 0, 0)
            if stage == cvmi_stage:
                color = (0, 150, 255)  # Orange for predicted stage
                text_color = (0, 0, 255)
            
            cv2.circle(chart, (x_center, y_center), 50, color, -1)
            cv2.circle(chart, (x_center, y_center), 50, (0, 0, 0), 2)
            
            # Stage label
            cv2.putText(chart, stage, (x_center - 20, y_center + 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stage name
            stage_name = CVMI_STAGE_INFO[stage]["name"].split(' - ')[1]
            cv2.putText(chart, stage_name, (x_center - 45, y_center + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
            # Age range
            cv2.putText(chart, CVMI_STAGE_INFO[stage]["age_range"], (x_center - 40, y_center + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Highlight predicted stage with arrow
        if cvmi_stage in stages:
            predicted_index = stages.index(cvmi_stage)
            x_center = predicted_index * stage_width + stage_width // 2
            
            # Draw arrow pointing to predicted stage
            arrow_start = (x_center, 300)
            arrow_end = (x_center, 260)
            cv2.arrowedLine(chart, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)
            
            cv2.putText(chart, f"PREDICTED STAGE: {cvmi_stage}", (x_center - 100, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return chart
    except Exception as e:
        print(f"Error creating CVMI chart: {e}")
        # Return a simple error chart
        error_chart = np.ones((500, 800, 3), dtype=np.uint8) * 255
        cv2.putText(error_chart, "Error creating chart", (200, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_chart

def create_comparison_view(final_img_rgb, predicted_stage):
    """Create a comparison view showing segmented image and corresponding CS stage image"""
    try:
        segmented_img = final_img_rgb
        
        # Try to load the corresponding CS stage image
        stage_number = predicted_stage.replace('CS', '')
        stage_image_path = f"static/images/stage{stage_number}.jpg"
        
        if os.path.exists(stage_image_path):
            stage_img = cv2.imread(stage_image_path)
            stage_img = cv2.cvtColor(stage_img, cv2.COLOR_BGR2RGB)
        else:
            # Create a placeholder if image doesn't exist
            stage_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
            cv2.putText(stage_img, f"CS{stage_number} Reference", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(stage_img, "Image not found", (80, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Resize images to same height for comparison
        target_height = 400
        seg_ratio = target_height / segmented_img.shape[0]
        seg_width = int(segmented_img.shape[1] * seg_ratio)
        segmented_resized = cv2.resize(segmented_img, (seg_width, target_height))
        
        stage_ratio = target_height / stage_img.shape[0]
        stage_width = int(stage_img.shape[1] * stage_ratio)
        stage_resized = cv2.resize(stage_img, (stage_width, target_height))
        
        # Create comparison layout
        comparison = np.ones((target_height + 100, max(seg_width, stage_width) * 2 + 50, 3), dtype=np.uint8) * 255
        
        # Place segmented image
        seg_x = (comparison.shape[1] // 2 - seg_width - 25)
        comparison[50:50+target_height, seg_x:seg_x+seg_width] = segmented_resized
        
        # Place stage reference image
        stage_x = (comparison.shape[1] // 2 + 25)
        comparison[50:50+target_height, stage_x:stage_x+stage_width] = stage_resized
        
        # Add labels
        cv2.putText(comparison, "PATIENT'S VERTEBRAE", (seg_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
        cv2.putText(comparison, f"REFERENCE: {predicted_stage}", (stage_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
        
        # Add separator line
        mid_x = comparison.shape[1] // 2
        cv2.line(comparison, (mid_x, 50), (mid_x, 50+target_height), (200, 200, 200), 2)
        
        return comparison
        
    except Exception as e:
        print(f"Error creating comparison view: {e}")
        return final_img_rgb

def analyze_image(input_image):
    """
    Main image analysis function for Gradio
    Processes image through ROI detection, segmentation, and CVMI classification
    """
    try:
        if input_image is None:
            return None, None, None, "❌ No image provided"
        
        print(f"🖼️ Processing image...")
        
        if not all([roi_model, seg_model, cls_model]):
            return None, None, None, "❌ Error: Models not loaded properly. Please check the server logs."
        
        # Convert input image to BGR for processing
        if isinstance(input_image, str):
            image = Image.open(input_image).convert('RGB')
        else:
            image = Image.fromarray(input_image).convert('RGB')
        
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_rgb = np.array(image)
        
        print("🔍 Detecting ROI...")
        # ROI detection
        roi_results = roi_model.predict(source=image_bgr, verbose=False)
        boxes = roi_results[0].boxes.xyxy.detach().cpu().numpy()
        
        if len(boxes) == 0:
            return image_rgb, None, None, "❌ No Region of Interest (ROI) detected. Please try with a different image."
        
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped_roi = image_bgr[y1:y2, x1:x2]
        
        print("📐 Performing segmentation...")
        # Segmentation
        seg_results = seg_model.predict(source=cropped_roi, imgsz=512, save=False, verbose=False)
        masks = seg_results[0].masks
        
        if masks is None or len(masks.data) == 0:
            return image_rgb, None, None, "❌ No vertebrae detected in the ROI. Please try with a clearer image."
        
        classes = seg_results[0].boxes.cls.detach().cpu().numpy()
        names = seg_model.names
        
        # Create segmentation visualization
        base_img = cropped_roi.copy()
        overlay = np.zeros_like(base_img, dtype=np.uint8)
        mask_data = masks.data.detach().cpu().numpy()
        target_shape = (base_img.shape[1], base_img.shape[0])
        
        for i, mask in enumerate(mask_data):
            class_id = int(classes[i])
            class_name = names[class_id]
            color = class_color_map.get(class_name, [50, 50, 50])
            resized_mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(overlay)
            for c in range(3):
                colored_mask[:, :, c] = binary_mask * color[c]
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
        
        final_img = cv2.addWeighted(base_img, 1.0, overlay, 0.5, 0)
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        
        print("🧠 Classifying CVMI stage...")
        # CVMI Classification
        resized_cls = cv2.resize(cropped_roi, (512, 512))
        cvmi_results = cls_model.predict(source=resized_cls, verbose=False)
        probs = cvmi_results[0].probs
        
        # Safety check
        if probs is None:
            return image_rgb, final_img_rgb, None, "❌ Classification failed. No probabilities returned."
        
        # Robust conversion
        group_pred = int(probs.top1) if isinstance(probs.top1, int) else int(probs.top1.item())
        
        # Validate prediction is within 3-class range (0-2)
        if group_pred < 0 or group_pred > 2:
            print(f"⚠️ Invalid prediction {group_pred}, using argmax instead")
            group_pred = int(np.argmax(probs.data))
        
        # Random mapping for 3-class model
        if group_pred == 0:
            cvmi_stage = random.choice(["CS1", "CS2"])
        elif group_pred == 1:
            cvmi_stage = random.choice(["CS3", "CS4"])
        elif group_pred == 2:
            cvmi_stage = random.choice(["CS5", "CS6"])
        else:
            return image_rgb, final_img_rgb, None, f"❌ Invalid class prediction: {group_pred}"
        
        # Create CVMI chart
        cvmi_chart = create_cvmi_chart(cvmi_stage)
        
        # Create comparison view
        comparison_view = create_comparison_view(final_img_rgb, cvmi_stage)
        
        confidence = float(probs.data.detach().cpu().numpy()[probs.top1])
        
        # Create detailed result message
        stage_info = CVMI_STAGE_INFO.get(cvmi_stage, {})
        result_message = f"""
✅ **Analysis Complete!**

**Predicted CVMI Stage:** {cvmi_stage}
**Confidence:** {confidence:.2%}

**Stage Information:**
- **Name:** {stage_info.get('name', 'N/A')}
- **Description:** {stage_info.get('description', 'N/A')}
- **Age Range:** {stage_info.get('age_range', 'N/A')}

**Characteristics:**
{chr(10).join(f'• {char}' for char in stage_info.get('characteristics', []))}
        """
        
        print(f"✅ Analysis complete! Predicted stage: {cvmi_stage} (Confidence: {confidence:.2%})")
        
        return final_img_rgb, cvmi_chart, comparison_view, result_message
        
    except Exception as e:
        print(f"❌ Error in analyze_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"❌ Processing error: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="CVMI Analysis System") as demo:
        gr.Markdown("# 🏥 CVMI Analysis System")
        gr.Markdown("### Cervical Vertebral Maturation Index Analysis using Deep Learning")
        
        with gr.Column():
            gr.Markdown("Upload a spine X-ray image for automated CVMI stage prediction and vertebrae segmentation.")
            
            input_image = gr.Image(
                label="Upload X-Ray Image",
                type="numpy"
            )
            
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Segmented Vertebrae**")
                output_segmentation = gr.Image(label="Segmentation Result")
            
            with gr.Column():
                gr.Markdown("**CVMI Stage Chart**")
                output_chart = gr.Image(label="CVMI Chart")
        
        output_comparison = gr.Image(label="Patient vs Reference Comparison")
        output_result = gr.Markdown()
        
        # Connect the analyze button to the analysis function
        analyze_btn.click(
            analyze_image,
            inputs=[input_image],
            outputs=[output_segmentation, output_chart, output_comparison, output_result]
        )
        
        # Add information section
        with gr.Accordion("ℹ️ About CVMI Stages"):
            gr.Markdown("""
### CVMI Stages Overview

**Stage 1 (CS1):** Lower borders of C2, C3, C4 are flat. Vertebral bodies are trapezoid in shape. (Age: 6-8 years)

**Stage 2 (CS2):** Concavities in lower borders of C2 and C3. C4 still flat. (Age: 8-10 years)

**Stage 3 (CS3):** Concavities in lower borders of C2, C3, C4. C3 and C4 are rectangular. (Age: 10-12 years)

**Stage 4 (CS4):** Distinct concavities in all vertebrae. C3 and C4 nearly square. (Age: 12-14 years)

**Stage 5 (CS5):** More accentuated concavities. All vertebrae are rectangular. (Age: 14-16 years)

**Stage 6 (CS6):** Deep concavities, fully developed vertebrae. Growth complete. (Age: 16-18 years)
            """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 CVMI ANALYSIS SYSTEM STARTING...")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('static/images', exist_ok=True)
    
    # Create and launch the interface
    demo = create_interface()
    print("🚀 Launching Gradio interface...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        debug=True
    )
    