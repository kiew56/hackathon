import flet as ft
import os
import cv2
import threading
from PIL import Image, ImageDraw
import io
import base64
import json
import flet_video as fv
import numpy as np
from ultralytics import YOLO

# Load your .pt model once (fast subsequent calls)
PT_MODEL_PATH = "/assets/best-5.pt"
model_pt = YOLO(PT_MODEL_PATH)

CLASS_NAMES = ["abrasions", "bruises", "snake_bites", "tick_bites", "burns", "cut", "ingrown_nail", "laceration"]

def run_inference_pt(image_path, conf_threshold=0.25, device="cpu"):
    """
    Returns tuple: (injury_type: str, confidence: float, bounding_box: list)
    Returns (None, 0.0, None) if no detection found
    """
    results = model_pt.predict(source=image_path, imgsz=640, conf=conf_threshold, device=device, verbose=False)
    
    if not results:
        return None, 0.0, None
    
    r = results[0]  # first (and only) frame
    
    # Check if there are any detections
    if not hasattr(r, "boxes") or len(r.boxes) == 0:
        return None, 0.0, None
    
    # Get the detection with highest confidence
    xyxy = r.boxes.xyxy.cpu().numpy()    # shape (N,4)
    confs = r.boxes.conf.cpu().numpy()   # shape (N,)
    clsids = r.boxes.cls.cpu().numpy()   # shape (N,)
    
    # Find the detection with maximum confidence
    max_conf_idx = np.argmax(confs)
    best_conf = float(confs[max_conf_idx])
    best_class_id = int(clsids[max_conf_idx])
    best_box = xyxy[max_conf_idx].tolist()
    
    # Get class name
    injury_type = CLASS_NAMES[best_class_id] if best_class_id < len(CLASS_NAMES) else "unknown"
    
    return injury_type, best_conf, best_box

# Load injury data from JSON
injury_data = {}
try:
    json_path = os.path.join(os.path.dirname(__file__), 'assets/description.json')
    with open(json_path) as f:
        injury_data = json.load(f)
except FileNotFoundError:
    print("Warning: description.json not found. Injury information will not be available.")

def main(page: ft.Page):
    page.title = "HealAI"
    page.window_width = 600
    page.window_height = 1000
    page.window_resizable = True
    page.bgcolor = ft.Colors.WHITE
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO

    # HEADER
    s = ft.Container(
        content=ft.Text(
            "HealAI",
            size=40, 
            weight=ft.FontWeight.BOLD, 
            color=ft.Colors.BLACK54
        )
    )
    T = ft.Container(
        content=ft.Text(
            "AI-powered injury scanner",
            size=20,
            color=ft.Colors.BLACK54,
        )
    )
    
    # VARIABLES
    I = ft.Image(
        src="/Users/kiewmthombeni/Documents/Injury app/my-app/src/assets/home.png",
        width=260,
        height=260,
    )

    # File picker for images
    file_picker = ft.FilePicker(
        on_result=lambda e: on_file_result(e)
    )
    page.overlay.append(file_picker)

    # Result display
    result_text = ft.Text("", size=14, color=ft.Colors.BLACK)
    result_container = ft.Container(
        content=result_text,
        visible=False,
        padding=20,
        bgcolor=ft.Colors.LIGHT_BLUE_50,
        border_radius=10,
        margin=20
    )

    # Processed image with bounding boxes
    processed_image = ft.Image(width=400, height=300, fit=ft.ImageFit.CONTAIN)

    # Result display for injury information
    injury_info_container = ft.Container(
        content=ft.Column(),
        visible=False,
        padding=20,
        bgcolor=ft.Colors.LIGHT_GREEN_50,
        border_radius=10,
        margin=20
    )

    # YouTube video embed
    youtube_container = ft.Container(
        content=ft.Column(),
        visible=False,
        padding=20,
        bgcolor=ft.Colors.LIGHT_BLUE,
        border_radius=10,
        margin=10
    )

    # HOME PAGE
    R = ft.Container(
        content=ft.Column(
            controls=[
                ft.Container(
                    content=ft.ElevatedButton(
                        "Scan a Video",
                        width=200,
                        height=50,
                        bgcolor=ft.Colors.BLUE,
                        color=ft.Colors.WHITE,
                        on_click=lambda e: switch_page("video")
                    )
                )
            ]
        )
    )

    home_page_content = ft.Column(
        controls=[
            ft.Container(content=I),
            ft.Container(content=R)
        ]
    )

    # VIDEO SCAN PAGE
    video_status = ft.Text("Camera ready", size=14, color=ft.Colors.GREY)
    camera_feed = ft.Image(width=640, height=480, fit=ft.ImageFit.CONTAIN)
    video_detection_text = ft.Text("", size=14, color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD)
    
    # Video page injury information display
    video_injury_info_container = ft.Container(
        content=ft.Column(),
        visible=False,
        padding=20,
        bgcolor=ft.Colors.LIGHT_GREEN_50,
        border_radius=10,
        margin=20,
        width=580
    )
    
    # Video page YouTube embed
    video_youtube_container = ft.Container(
        content=ft.Column(),
        visible=False,
        padding=20,
        bgcolor=ft.Colors.LIGHT_BLUE_50,
        border_radius=10,
        margin=20,
        width=580
    )
    
    camera_active = False
    cap = None
    last_detected_injury = None  # Track last shown injury to avoid repeated updates
    
    def start_camera():
        nonlocal camera_active, cap, last_detected_injury
        if camera_active:
            return  # Already running
        
        camera_active = True
        cap = cv2.VideoCapture(0)
        last_detected_injury = None  # Reset when camera starts
        
        if not cap.isOpened():
            video_status.value = "Error: Could not open camera"
            camera_active = False
            page.update()
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        video_status.value = "Camera running - Real-time detection active"
        page.update()
        
        def capture_frames():
            nonlocal last_detected_injury
            frame_count = 0
            frames_since_detection = 0  # Counter for stable detection
            
            while camera_active:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                original_frame = frame.copy()
                
                # Run inference every frame (you can change to every N frames for performance)
                # For better performance, run every 5 frames: if frame_count % 5 == 0:
                try:
                    # Save frame temporarily for YOLO inference
                    temp_path = "temp_frame.jpg"
                    cv2.imwrite(temp_path, original_frame)
                    
                    # Run YOLO detection
                    results = model_pt.predict(source=temp_path, imgsz=640, conf=0.35, device="cpu", verbose=False)
                    
                    detection_found = False
                    highest_conf_injury = None
                    highest_conf = 0.0
                    
                    if results and len(results) > 0:
                        r = results[0]
                        if hasattr(r, "boxes") and len(r.boxes) > 0:
                            detection_found = True
                            xyxy = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            clsids = r.boxes.cls.cpu().numpy()
                            
                            # Find highest confidence detection
                            max_idx = np.argmax(confs)
                            highest_conf = float(confs[max_idx])
                            highest_conf_injury = CLASS_NAMES[int(clsids[max_idx])] if int(clsids[max_idx]) < len(CLASS_NAMES) else None
                            
                            # Draw all detections on the frame
                            for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clsids)):
                                x1, y1, x2, y2 = map(int, box)
                                class_id = int(cls_id)
                                confidence = float(conf)
                                
                                # Get class name
                                injury_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
                                
                                # Draw bounding box (highlight highest confidence in different color)
                                if i == max_idx:
                                    color = (255, 0, 0)  # Blue (BGR) for highest confidence
                                    thickness = 4
                                else:
                                    color = (0, 255, 0)  # Green for others
                                    thickness = 3
                                
                                cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Create label with background
                                label = f"{injury_name}: {confidence:.2f}"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                text_thickness = 2
                                
                                # Get text size for background
                                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                                
                                # Draw background rectangle for text
                                cv2.rectangle(original_frame, 
                                            (x1, y1 - text_height - 10), 
                                            (x1 + text_width, y1), 
                                            color, -1)
                                
                                # Draw text
                                cv2.putText(original_frame, label, 
                                          (x1, y1 - 5), 
                                          font, font_scale, (255, 255, 255), text_thickness)
                            
                            # Update status text with detection info
                            if len(xyxy) == 1:
                                video_detection_text.value = f"🔍 Detected: {highest_conf_injury} ({highest_conf:.2%})"
                            else:
                                video_detection_text.value = f"🔍 Detected {len(xyxy)} injuries - Primary: {highest_conf_injury} ({highest_conf:.2%})"
                            
                            # Load injury information if confidence is high enough and it's a new detection
                            if highest_conf >= 0.4 and highest_conf_injury and highest_conf_injury != last_detected_injury:
                                frames_since_detection += 1
                                # Wait for 3 consecutive frames with same detection for stability
                                if frames_since_detection >= 3:
                                    last_detected_injury = highest_conf_injury
                                    frames_since_detection = 0
                                    load_injury_info_video(highest_conf_injury, highest_conf)
                            elif highest_conf_injury != last_detected_injury:
                                frames_since_detection = 0
                    
                    if not detection_found:
                        video_detection_text.value = "👁️ No injuries detected - Show injury to camera"
                        frames_since_detection = 0
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                except Exception as e:
                    print(f"Detection error: {e}")
                    video_detection_text.value = f"⚠️ Detection error: {str(e)}"
                
                # Convert frame to RGB for display
                frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                img_bytes = io.BytesIO()
                pil_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                camera_feed.src_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                page.update()
        
        thread = threading.Thread(target=capture_frames, daemon=True)
        thread.start()
    
    def load_injury_info_video(injury_type, confidence):
        """Load injury information and video for detected injury in video mode"""
        injury_info = injury_data.get(injury_type)
        if injury_info:
            video_injury_info_container.content = ft.Column([
                ft.Text(f"📋 {injury_info.get('title', injury_type.title())}", 
                       size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
                ft.Divider(height=2, color=ft.Colors.BLACK),
                
                ft.Text("Description", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_900),
                ft.Text(injury_info.get("description", "No description available."), color=ft.Colors.BLACK, size=13),
                
                ft.Text("Symptoms", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.ORANGE_900),
                ft.Text("• " + "\n• ".join(injury_info.get("symptoms", [])), color=ft.Colors.BLACK, size=12),
                
                ft.Text("First Aid Steps", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_900),
                ft.Column([ft.Text(f"{i+1}. {step}", color=ft.Colors.BLACK, size=12) 
                          for i, step in enumerate(injury_info.get("first_aid_steps", []))]),
                
                ft.Text("⚠️ When to Seek Medical Help", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_700),
                ft.Column([ft.Text(f"• {warning}", size=12, color=ft.Colors.RED_800) 
                          for warning in injury_info.get("when_to_seek_help", [])])
            ], scroll=ft.ScrollMode.AUTO, height=400)
            video_injury_info_container.visible = True
            
            # Embed YouTube video
            video_url = injury_info.get("video_url")

# Extract YouTube video ID safely
            

            sample_media = [
                fv.VideoMedia(video_url)
            ]

            video_youtube_container.content = ft.Column([
                ft.Text(
                    "🎥 First Aid Video Guide",
                    size=16,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Container(
                    fv.Video(
                        expand=True,
                        playlist=sample_media,
                        playlist_mode=fv.PlaylistMode.LOOP,
                        fill_color=ft.Colors.BLUE_400,
                        aspect_ratio=16/9,
                        volume=100,
                        autoplay=False,
                        filter_quality=ft.FilterQuality.HIGH,
                        muted=False,
                        on_loaded=lambda e: print("Video loaded successfully!"),
                        on_enter_fullscreen=lambda e: print("Video entered fullscreen!"),
                        on_exit_fullscreen=lambda e: print("Video exited fullscreen!"),
                    )
                ),
            ])

            video_youtube_container.visible = True
        else:
             # URL exists but is not a valid YouTube URL
            video_youtube_container.visible = False
        page.update()
    
    def stop_camera():
        nonlocal camera_active, cap, last_detected_injury
        camera_active = False
        if cap:
            cap.release()
            cap = None
        last_detected_injury = None
        video_status.value = "Camera stopped"
        video_injury_info_container.visible = False
        video_youtube_container.visible = False
        page.update()
    
    video_page_content = ft.Column(
        controls=[
            ft.Text("🎥 Real-Time Injury Scanner", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK54),
            camera_feed,
            video_detection_text,
            video_status,
            ft.Row(
                controls=[
                    ft.ElevatedButton(
                        "▶ Start Camera",
                        width=140,
                        height=45,
                        bgcolor=ft.Colors.GREEN,
                        color=ft.Colors.WHITE,
                        on_click=lambda e: start_camera()
                    ),
                    ft.ElevatedButton(
                        "⏹ Stop Camera",
                        width=140,
                        height=45,
                        bgcolor=ft.Colors.RED,
                        color=ft.Colors.WHITE,
                        on_click=lambda e: stop_camera()
                    )
                ],
                spacing=10
            ),
            video_injury_info_container,
            video_youtube_container,
            ft.ElevatedButton(
                "🏠 Back to Home",
                width=150,
                height=40,
                bgcolor=ft.Colors.GREY_700,
                color=ft.Colors.WHITE,
                on_click=lambda e: switch_page("home")
            )
        ],
        spacing=20,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO
    )

    # IMAGE SCAN PAGE
    image_status = ft.Text("No image selected", size=14, color=ft.Colors.GREY)
    
   
    # File picker result handler
    def on_file_result(e):
        if not e.files:
            return
        
        # macOS fix: check if path exists, fallback to name
        file_obj = e.files[0]
        file_path = file_obj.path
        
        # On some systems, path may be None; try to construct it
        if not file_path or file_path is None:
            image_status.value = "Error: Could not access file path"
            page.update()
            return
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            image_status.value = "Please select a valid image file (.jpg, .jpeg, .png)"
            page.update()
            return
        
        image_status.value = f"Processing: {file_name}..."
        page.update()
        
        try:
            # Run model inference
            injury_type, confidence, bounding_box = run_inference_pt(file_path)
            
            if injury_type:
                result_text.value = f"Identified: {injury_type}\nConfidence: {confidence:.2%}"
                result_container.visible = True
                
                # Display processed image with bounding box
                img = Image.open(file_path).convert('RGB')
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)
                
                # Draw actual bounding box from detection
                if bounding_box:
                    x1, y1, x2, y2 = map(int, bounding_box)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                    
                    # Add label
                    label = f"{injury_type} ({confidence:.2%})"
                    draw.text((x1, y1 - 10), label, fill="red")
                
                img_bytes = io.BytesIO()
                draw_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                processed_image.src_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                
                # Fetch and display injury information
                injury_info = injury_data.get(injury_type)
                if injury_info:
                    injury_info_container.content = ft.Column([
                        ft.Text(injury_info.get("title", injury_type.title()), size=20, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text("Description", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text(injury_info.get("description", "No description available."), size=12),
                        ft.Text("Symptoms", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text(", ".join(injury_info.get("symptoms", [])), size=12),
                        ft.Text("First Aid Steps", size=16, weight=ft.FontWeight.BOLD),
                        ft.Column([ft.Text(f"• {step}", size=12) for step in injury_info.get("first_aid_steps", [])]),
                        ft.Text("When to Seek Help", size=16, weight=ft.FontWeight.BOLD),
                        ft.Column([ft.Text(f"• {warning}", size=12) for warning in injury_info.get("when_to_seek_help", [])])
                    ], scroll=ft.ScrollMode.AUTO)
                    injury_info_container.visible = True
                    
                    # Embed YouTube video
                    video_url = injury_info.get("video_url")
                    if video_url and "youtube.com" in video_url:
                        video_id = video_url.split("v=")[-1].split("&")[0]
                        
                        youtube_container.content = ft.Column([
                            ft.Text("Video Guide", size=16, weight=ft.FontWeight.BOLD),
                            ft.Container(
                                content=ft.Image(
                                    src=f"https://img.youtube.com/vi/{video_id}/0.jpg",
                                    width=350,
                                    height=200,
                                    fit=ft.ImageFit.COVER
                                ),
                                on_click=lambda e: page.launch_url(video_url)
                            ),
                            ft.Text("Click on the image to watch the video", size=10, italic=True)
                        ])
                        youtube_container.visible = True
                    else:
                        youtube_container.visible = False
                else:
                    injury_info_container.visible = False
                    youtube_container.visible = False
                
                image_status.value = f"Completed: {file_name}"
            else:
                result_text.value = "No injury detected in this image"
                result_container.visible = True
                injury_info_container.visible = False
                youtube_container.visible = False
                
                # Show original image
                img = Image.open(file_path).convert('RGB')
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                processed_image.src_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                
                image_status.value = f"No detection: {file_name}"
        
        except Exception as ex:
            result_text.value = f"Error processing image: {str(ex)}"
            result_container.visible = True
            injury_info_container.visible = False
            youtube_container.visible = False
            image_status.value = f"Error: {file_name}"
            print(f"Full error: {ex}")  # For debugging
        
        page.update()

    def switch_page(page_name):
        home_page_content.visible = page_name == "home"
        video_page_content.visible = page_name == "video"
        
        
        if page_name != "video":
            stop_camera()
        
        # Reset visibility of result containers when switching pages
        if page_name == "home":
            result_container.visible = False
            injury_info_container.visible = False
            youtube_container.visible = False
            image_status.value = "No image selected"
        
        page.update()

    page.add(s, T, home_page_content, video_page_content)
    switch_page("home")
    page.update()

ft.app(main)
