# Fixed version of the scene frame extraction and copying logic

from tqdm.notebook import tqdm
import shutil
import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.scene_detection import SceneDetector

# Create scene xlsx folder
scene_xlsx_folder = os.path.join(output_folder, "scene_xlsx")
os.makedirs(scene_xlsx_folder, exist_ok=True)

# Create scene metadata file path
scene_meta_name = "_".join([project_title, "video_scene_meta.xlsx"])
scene_meta_path = os.path.join(project_folder, scene_meta_name)

results = []
for video_path in tqdm(video_paths):
    video_name = os.path.basename(video_path)
    video_name_clean = os.path.splitext(video_name)[0]
    
    # Create output folder for this video FIRST
    output_folder_video = os.path.join(output_folder, video_name_clean)
    os.makedirs(output_folder_video, exist_ok=True)
    
    # Create temporary directory if it doesn't exist
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        print(f"Processing video: {video_name}")
        
        # Get video format and fps with error handling
        try:
            video_format, fps = get_video_format(video_path)
            print(f"Video format: {video_format}, FPS: {fps}")
        except Exception as e:
            print(f"Error getting video format: {e}")
            video_format, fps = "horizontal", 25  # Default fallback
            
        # Extract frames with error handling
        try:
            frame_paths = get_frames(video_path, tmp_dir, video_format, max_image_length=image_length_max)
            print(f"Extracted {len(frame_paths)} frames to temporary directory")
            
            if len(frame_paths) == 0:
                raise ValueError("No frames were extracted from the video")
                
        except Exception as e:
            print(f"Error extracting frames: {e}")
            results.append([video_path, video_name, None, np.nan, np.nan, f"Frame extraction failed: {str(e)}"])
            continue
        
        # Detect scenes with error handling
        try:
            scene_detector = SceneDetector(video_path=video_path, frame_paths=frame_paths, fps=fps, debug_handler=None, verbose=False)
            scene_df = scene_detector.detect_scenes(n_scene_frames=n_scene_frames, embedder=embedder)
            
            if scene_df is None or len(scene_df) == 0:
                raise ValueError("No scenes detected in the video")
                
            print(f"Scene detection completed. Found {len(scene_df)} scene frames")
            
        except Exception as e:
            print(f"Error in scene detection: {e}")
            # Clean up temporary files
            for f in frame_paths:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            results.append([video_path, video_name, None, np.nan, np.nan, f"Scene detection failed: {str(e)}"])
            continue
        
        # Process scene dataframe
        scene_df["scene_frame_id"] = scene_df["scene_image_path"].apply(lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        unique_scenes = scene_df["scene_number"].unique().tolist()
        print(f"Found {len(unique_scenes)} unique scenes with {len(scene_df)} total scene frames")

        # Add video metadata to scene dataframe
        scene_df["video_path"] = video_path
        scene_df["video_name"] = video_name_clean
        
        # COPY SCENE FRAMES TO OUTPUT FOLDER BEFORE CLEANUP
        scene_frames = scene_df["scene_image_path"].tolist()
        copied_frame_paths = []
        
        for i, scene_frame in enumerate(scene_frames):
            if os.path.exists(scene_frame):
                # Create meaningful filename with scene number and frame number
                scene_num = scene_df.iloc[i]["scene_number"]
                scene_frame_name = f"{video_name_clean}_scene_{scene_num:03d}_frame_{i+1:03d}.jpg"
                scene_frame_output_path = os.path.join(output_folder_video, scene_frame_name)
                
                try:
                    shutil.copy2(scene_frame, scene_frame_output_path)
                    copied_frame_paths.append(scene_frame_output_path)
                    print(f"Copied: {scene_frame_name}")
                except Exception as copy_error:
                    print(f"Error copying {scene_frame}: {copy_error}")
            else:
                print(f"Warning: Scene frame not found: {scene_frame}")
        
        print(f"Successfully copied {len(copied_frame_paths)} scene frames to {output_folder_video}")
        
        # Update scene_df with new paths for Excel file
        scene_df_for_excel = scene_df.copy()
        if len(copied_frame_paths) >= len(scene_df_for_excel):
            scene_df_for_excel["output_image_path"] = copied_frame_paths[:len(scene_df_for_excel)]
        else:
            # Handle case where fewer files were copied than expected
            scene_df_for_excel["output_image_path"] = copied_frame_paths + [None] * (len(scene_df_for_excel) - len(copied_frame_paths))
        
        # Save scene dataframe to Excel
        scene_xlsx_name = os.path.join(scene_xlsx_folder, video_name_clean + ".xlsx")
        try:
            scene_df_for_excel.to_excel(scene_xlsx_name, index=False)
            print(f"Saved scene data to: {scene_xlsx_name}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")
            scene_xlsx_name = None
        
        # Clean up temporary frame files AFTER copying
        cleaned_files = 0
        for f in frame_paths:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    cleaned_files += 1
                except Exception as e:
                    print(f"Error removing temporary file {f}: {e}")
        print(f"Cleaned up {cleaned_files} temporary files")
        
        num_unique_scenes = len(unique_scenes)
        num_scene_frames = len(scene_df)
        message = f"Success - copied {len(copied_frame_paths)} images"
        
    except Exception as e:
        scene_xlsx_name = None
        num_unique_scenes = np.nan
        num_scene_frames = np.nan
        message = str(e)
        print(f"!!!!!    Error processing {video_name}: {message}")
        
        # Clean up temporary files even on error
        try:
            if 'frame_paths' in locals():
                for f in frame_paths:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass
        except:
            pass

    results.append([video_path, video_name, scene_xlsx_name, num_unique_scenes, num_scene_frames, message])

# Create and save results dataframe
result_df = pd.DataFrame(results, columns=["video_path", "video_name", "scene_xlsx_path", "num_scenes", "num_scene_frames", "status"])
try:
    result_df.to_excel(scene_meta_path, index=False)
    print(f"Results saved to: {scene_meta_path}")
except Exception as e:
    print(f"Error saving results Excel file: {e}")

print(f"\nProcessing complete! Check the following locations:")
print(f"- Scene images: {output_folder}")
print(f"- Scene Excel files: {scene_xlsx_folder}")
print(f"- Summary report: {scene_meta_path}")

# Display results
result_df.head()