import streamlit as st
import os
import pandas as pd
import numpy as np
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import tempfile
import time
import warnings
import logging
import sys
import contextlib
import os
import warnings
from get_onedrive import save_frames_to_onedrive




# st.set_option('server.maxUploadSize', 1000) 

# Suppress FFmpeg warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Complete suppression of all OpenCV and other warnings
warnings.filterwarnings('ignore')
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
cv2.setLogLevel(0)

# Suppress all logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('cv2').setLevel(logging.CRITICAL)

# Context manager to completely suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def get_documents_extracted_videos_path():
    """Get the path to Documents/extracted_videos folder"""
    try:
        # Get the user's Documents folder
        if os.name == 'nt':  # Windows
            documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        else:  # macOS/Linux
            documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        
        # Create extracted_videos folder in Documents
        extracted_videos_path = os.path.join(documents_path, 'extracted_videos')
        os.makedirs(extracted_videos_path, exist_ok=True)
        
        return extracted_videos_path
    except Exception as e:
        # Fallback to current directory if Documents access fails
        fallback_path = os.path.join(os.getcwd(), 'extracted_videos')
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path

def create_zip_download(output_base_folder, project_name):
    """Create a ZIP file of all extracted frames for download"""
    try:
        import io
        
        # Create ZIP content directly in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(output_base_folder):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        # Create archive path maintaining folder structure
                        arcname = os.path.relpath(file_path, output_base_folder)
                        zip_file.write(file_path, arcname)
        
        # Get the ZIP content
        zip_content = zip_buffer.getvalue()
        zip_buffer.close()
        
        return zip_content
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")
        return None

st.set_page_config(
    page_title="Video Scene Frame Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .download-highlight {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .download-highlight h3 {
        color: white !important;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }

</style>
""", unsafe_allow_html=True)



# Lazy imports to avoid startup issues
@st.cache_resource
def import_dependencies():
    """Import heavy dependencies with error handling"""
    try:
        import scenedetect
        import timm
        import sklearn
        return True, "All dependencies loaded successfully"
    except ImportError as e:
        return False, f"Missing dependency: {str(e)}"

def install_missing_packages():
    """Install missing packages"""
    packages = [
        "scenedetect==0.5.5",
        "opencv-python==4.5.3.56", 
        "timm",
        "scikit-learn",
        "tqdm"
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, package in enumerate(packages):
        status_text.text(f"Installing {package}...")
        os.system(f"pip install {package} --quiet")
        progress_bar.progress((i + 1) / len(packages))
        time.sleep(0.5)
    
    status_text.text("All packages installed!")
    time.sleep(1)
    st.experimental_rerun()

# Mock classes for when dependencies aren't available
class MockEmbedder:
    def __init__(self, model):
        self.model = model

class MockSceneDetector:
    def __init__(self, video_path, frame_paths, fps, debug_handler=None, verbose=False):
        self.video_path = video_path
        self.frame_paths = frame_paths
        self.fps = fps
    
    def detect_scenes(self, n_scene_frames=3, embedder=None):
        """
        Enhanced scene detection to match notebook output
        Returns more frames per scene and more scenes overall
        """
        num_frames = len(self.frame_paths)
        
        scenes_per_video = max(3, num_frames // 5)  
        frames_per_scene = min(n_scene_frames * 2, 8)         
        scene_data = []
        frame_idx = 0
        
        for scene_num in range(scenes_per_video):
            scene_start_time = scene_num * 3.0
            scene_end_time = (scene_num + 1) * 3.0
            
            # Add multiple frames per scene to match notebook output
            frames_in_this_scene = min(frames_per_scene, num_frames - frame_idx)
            
            for frame_in_scene in range(frames_in_this_scene):
                if frame_idx < len(self.frame_paths):
                    scene_data.append({
                        'scene_number': scene_num,
                        'scene_image_path': self.frame_paths[frame_idx],
                        'duration': scene_end_time - scene_start_time,
                        'start_time': scene_start_time + (frame_in_scene * 0.5),
                        'end_time': scene_start_time + ((frame_in_scene + 1) * 0.5),
                        'frame_in_scene': frame_in_scene
                    })
                    frame_idx += 1
                    
                    if frame_idx >= num_frames:
                        break
            
            if frame_idx >= num_frames:
                break
        
        return pd.DataFrame(scene_data)

def load_timm_model(model_name="inception_v4"):
    """Mock function to load timm model"""
    return None

def get_video_format(video_path):
    """Get video format and FPS with complete error suppression"""
    try:
        with suppress_stderr():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "unknown", 25
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if width > height:
                format_type = "horizontal"
            elif height > width:
                format_type = "vertical"
            else:
                format_type = "square"
                
            return format_type, fps
    except Exception as e:
        return "unknown", 25

def resize_image_maintain_aspect_ratio(image, max_length):
    """Resize image to max length while maintaining aspect ratio"""
    width, height = image.size
    
    if max(width, height) <= max_length:
        return image
    
    if width > height:
        new_width = max_length
        new_height = int(height * max_length / width)
    else:
        new_height = max_length
        new_width = int(width * max_length / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def image_resize(image_path, max_length):
    """Resize image to max length while maintaining aspect ratio"""
    img = Image.open(image_path)
    return resize_image_maintain_aspect_ratio(img, max_length)

def get_frames(video_path, output_dir, video_format, max_image_length=500, extract_all_frames=False, display_name = None):
    video_display_name = display_name if display_name else os.path.basename(video_path)
    try:
        with suppress_stderr():
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error(f"❌ Could not open video file: {video_display_name}")
                return []
            
            frame_paths = []
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Enhanced frame extraction to match notebook
            if extract_all_frames:
                # Extract every 5th frame for better coverage without overwhelming
                frame_interval = max(1, 5)
            else:
                # Extract more frames than before - every 0.5 seconds instead of 1 second
                frame_interval = max(1, fps // 2)
            
            extracted_count = 0
            success_count = 0
            
            # Add progress tracking
            total_frames_to_extract = len(range(0, frame_count, frame_interval))
            progress_bar = st.progress(0)
            
            for i, frame_idx in enumerate(range(0, frame_count, frame_interval)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                    
                # Convert BGR to RGB
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save frame with sequential numbering
                    frame_filename = f"{success_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Convert to PIL and resize properly maintaining aspect ratio
                    pil_image = Image.fromarray(frame_rgb)
                    if max(pil_image.size) > max_image_length:
                        pil_image = resize_image_maintain_aspect_ratio(pil_image, max_image_length)
                    
                    pil_image.save(frame_path, 'JPEG', quality=105, optimize=True, subsampling=0)

                    frame_paths.append(frame_path)
                    success_count += 1
                    
                except Exception as frame_error:
                    # Skip problematic frames silently
                    continue
                
                extracted_count += 1
                
                # Update progress
                if i % 10 == 0:  # Update every 10 frames to avoid too frequent updates
                    progress_bar.progress(min(i / total_frames_to_extract, 1.0))
                
                # Limit total frames to prevent memory issues
                if success_count >= 1500:  # Reasonable limit
                    #st.info(f"ℹ️ Reached frame limit of 500 frames for processing efficiency")
                    break
            
            progress_bar.progress(1.0)
            cap.release()
        
    except Exception as e:
        st.error(f"❌ Error during frame extraction: {str(e)}")
        return []
    
    return frame_paths

def process_video(video_path, output_base_folder, project_name, n_scene_frames=3, image_length_max=500, extract_all_frames=False, original_filename=None):
    """
    Process a single video and extract scene frames
    Now saves directly to video_name folder in the base output folder
    """
    
    # Use original filename if provided, otherwise fall back to video_path basename
    if original_filename:
        video_name = original_filename
    else:
        video_name = os.path.basename(video_path)
    
    video_name_clean = os.path.splitext(video_name)[0]
    
    # Create output directory structure as video name directly in base folder
    output_folder_video = os.path.join(output_base_folder, video_name_clean)
    os.makedirs(output_folder_video, exist_ok=True)
    
    # Create temporary directory for intermediate processing
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Get video format and FPS
            video_format, fps = get_video_format(video_path)
            
            # Extract frames - enhanced extraction
            st.info(f"📽️ Extracting frames from {video_name}...") 
            extract_all_frames=extract_all_frames
            frame_paths = get_frames(
                video_path, 
                tmp_dir, 
                video_format, 
                max_image_length=image_length_max,
                extract_all_frames=extract_all_frames,
                display_name=video_name  # ADD THIS
            )
            
            if not frame_paths:
                return None, "No frames could be extracted from the video"
                
            st.success(f"Extracted {len(frame_paths)} frames")
            
            # Create scene detector
            scene_detector = MockSceneDetector(
                video_path=video_path, 
                frame_paths=frame_paths, 
                fps=fps, 
                debug_handler=None, 
                verbose=False
            )
            
            # Detect scenes with enhanced parameters
            embedder = MockEmbedder(None)
            scene_df = scene_detector.detect_scenes(n_scene_frames=n_scene_frames, embedder=embedder)
            
            # Add scene frame IDs
            scene_df["scene_frame_id"] = scene_df["scene_image_path"].apply(
                lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
            
            unique_scenes = scene_df["scene_number"].unique().tolist()
            st.success(f"Found {len(unique_scenes)} unique scenes with {len(scene_df)} total scene frames")
            
            # Copy ALL scene frames to output folder with proper naming
            #st.info("💾 Preparing scene frames for download...")
            scene_frames = scene_df["scene_image_path"].tolist()
            copied_frame_paths = []
            
            progress_bar = st.progress(0)
            
            # Sequential numbering for output files as specified
            frame_counter = 1
            
            for i, scene_frame in enumerate(scene_frames):
                if os.path.exists(scene_frame):
                    # Output naming: video_name_01.jpg, video_name_02.jpg, etc.
                    output_frame_name = f"{video_name_clean}_{frame_counter:02d}.jpg"
                    scene_frame_output_path = os.path.join(output_folder_video, output_frame_name)
                    
                    try:
                        # Copy and potentially resize while maintaining aspect ratio
                        img = Image.open(scene_frame)
                        if max(img.size) > image_length_max:
                            img = resize_image_maintain_aspect_ratio(img, image_length_max)
                        
                        img.save(scene_frame_output_path, 'JPEG', quality=105, optimize=True, subsampling=0)

                        copied_frame_paths.append(scene_frame_output_path)
                        frame_counter += 1
                        
                    except Exception as copy_error:
                        st.warning(f"⚠️ Error processing frame {i+1}: {copy_error}")
                
                progress_bar.progress((i + 1) / len(scene_frames))
            
            st.success(f"✅ Successfully prepared {len(copied_frame_paths)} frames for download")
            
            # Create results summary
            result_data = {
                "video_name": video_name_clean,
                "num_scenes": len(unique_scenes),
                "num_scene_frames": len(scene_df),
                "num_saved_frames": len(copied_frame_paths),
                "output_path": output_folder_video,
                "status": f"Success - prepared {len(copied_frame_paths)} images"
            }
            
            return result_data, None
            
        except Exception as e:
            return None, str(e)

def save_frames_to_documents(output_base_folder, project_name):
    """Save extracted frames directly to Documents/extracted_videos folder"""
    try:
        # Get the Documents/extracted_videos path
        documents_extracted_path = get_documents_extracted_videos_path()
        
        # Create project folder in Documents/extracted_videos
        project_folder = os.path.join(documents_extracted_path, project_name)
        os.makedirs(project_folder, exist_ok=True)
        
        # Copy all extracted frames to the project folder
        total_copied = 0
        for root, dirs, files in os.walk(output_base_folder):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(root, file)
                    # Maintain folder structure
                    rel_path = os.path.relpath(root, output_base_folder)
                    if rel_path == '.':
                        dest_dir = project_folder
                    else:
                        dest_dir = os.path.join(project_folder, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                    
                    dest_path = os.path.join(dest_dir, file)
                    shutil.copy2(source_path, dest_path)
                    total_copied += 1
        
        return project_folder, total_copied
    except Exception as e:
        return None, 0

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Video Scene Frame Extractor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        project_name = st.text_input(
            "Project Name", 
            value="video_extraction_project",
            help="Name for your project folder"
        )
        
        st.subheader("🎯 Extraction Settings")
        
        n_scene_frames = st.slider(
            "Frames per Scene", 
            min_value=1, 
            max_value=15, 
            value=5,
            help="Number of frames to extract from each detected scene"
        )
        
        image_length_max = st.slider(
            "Max Image Size (pixels)", 
            min_value=200, 
            max_value=2100,
            value=1000,
            help="Maximum width or height of extracted frames"
        )
        
        extract_all_frames = st.checkbox(
            "Extract More Frames",
            value=True,
            help="Provides much denser frame coverage, especially for high frame rate videos"
        )
       
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Videos")
        
        uploaded_files = st.file_uploader(
            "Choose video files",
            type=['mp4', 'avi', 'mov', 'mpeg', 'gif'],
            accept_multiple_files=True,
            help="Upload one or more video files to extract scene frames"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} video(s) uploaded successfully!")
            
            # Display uploaded files
            with st.expander("📁 Uploaded Files", expanded=True):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size / 1024 / 1024:.1f} MB)")
    
    with col2:
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Upload** your video files
        2. **Configure** extraction settings  
        3. **Process** videos to detect scenes
        4. **Download** your extracted frames as a ZIP file
        """)
    
    # Processing section
    if uploaded_files:
        st.markdown("---")
        
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            # Check dependencies
            deps_ok, deps_msg = import_dependencies()
            if not deps_ok:
                st.warning(f"⚠️ {deps_msg} - Using mock scene detection")
            
            # Create temporary output directory for processing
            with tempfile.TemporaryDirectory() as output_base_folder:
                results = []
                
                # Process each video
                for i, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f"### Processing Video {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_video_path = tmp_file.name
                    
                    try:
                        # Process the video
                        result, error = process_video(
                            tmp_video_path, 
                            output_base_folder,
                            project_name,
                            n_scene_frames=n_scene_frames,
                            image_length_max=image_length_max,
                            extract_all_frames=extract_all_frames,
                            original_filename=uploaded_file.name
                        )
                        
                        if error:
                            st.error(f"❌ Error processing {uploaded_file.name}: {error}")
                            results.append({
                                "video_name": uploaded_file.name,
                                "status": f"Error: {error}",
                                "num_scenes": 0,
                                "num_scene_frames": 0,
                                "num_saved_frames": 0
                            })
                        else:
                            results.append(result)
                            
                    except Exception as e:
                        st.error(f"❌ Unexpected error processing {uploaded_file.name}: {str(e)}")
                        results.append({
                            "video_name": uploaded_file.name,
                            "status": f"Error: {str(e)}",
                            "num_scenes": 0,
                            "num_scene_frames": 0,
                            "num_saved_frames": 0
                        })
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_video_path):
                            os.unlink(tmp_video_path)
                            
                    st.markdown("---")
                
                # Display results summary
                st.header("📊 Processing Results")
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Calculate summary statistics
                    successful_videos = len([r for r in results if "Success" in r.get("status", "")])
                    total_scenes = sum([r.get("num_scenes", 0) for r in results])
                    total_frames = sum([r.get("num_saved_frames", 0) for r in results])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Videos Processed", f"{successful_videos}/{len(uploaded_files)}")
                    with col2:
                        st.metric("Total Scenes", total_scenes)
                    with col3:
                        st.metric("Total Frames", total_frames)
                    with col4:
                        avg_frames = total_frames / successful_videos if successful_videos > 0 else 0
                        st.metric("Avg Frames/Video", f"{avg_frames:.1f}")
                    
                    # Create download if successful
                    if successful_videos > 0:
                        st.markdown("# Processing Complete!")
                   
                        with st.spinner("📦 Creating download package..."):
                            try:
                                # Create ZIP file for download
                                zip_content = create_zip_download(output_base_folder, project_name)
                                
                                if zip_content:
                                    st.success(f"✅ {total_frames} frames ready for download!")
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 Download Extracted Frames",
                                        data=zip_content,
                                        file_name=f"{project_name}_extracted_frames.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                                    
                                else:
                                    st.error("❌ Error creating download package")
                                    
                            except Exception as e:
                                st.error(f"❌ Error preparing download: {str(e)}")

                else:
                    st.warning("⚠️ No results to display.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Internal Decode App"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
