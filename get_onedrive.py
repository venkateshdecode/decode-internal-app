import os
import shutil
import platform
from pathlib import Path



def get_onedrive_documents_path():
    """
    Detects the user's OneDrive 'Dokumente' folder path.
    Returns:
        Path to OneDrive/Dokumente/extracted_videos or None if not found.
    """
    system = platform.system()

    # Windows OneDrive detection
    if system == "Windows":
        path = os.environ.get("OneDriveCommercial") or os.environ.get("OneDrive")
        if path:
            return Path(path) / "Dokumente" / "extracted_videos"

    # macOS OneDrive detection
    elif system == "Darwin":
        username = os.getlogin()
        potential_paths = [
            Path(f"/Users/{username}/Library/CloudStorage/OneDrive-{username}/Dokumente"),
            Path(f"/Users/{username}/OneDrive/Dokumente")
        ]
        for p in potential_paths:
            if p.exists():
                return p / "extracted_videos"

    return None


def save_frames_to_onedrive(output_base_folder, project_name):
    """
    Copies extracted image frames into OneDrive/Dokumente/extracted_videos/project_name.
    
    Args:
        output_base_folder (str): Folder containing extracted image frames.
        project_name (str): Name of the folder to create inside extracted_videos.

    Returns:
        Tuple[str or None, int]: (destination folder path as string, number of files copied)
    """
    try:
        onedrive_extracted_path = get_onedrive_documents_path()
        if not onedrive_extracted_path:
            raise FileNotFoundError("OneDrive 'Dokumente' folder not found.")

        # Create project folder
        project_folder = onedrive_extracted_path / project_name
        os.makedirs(project_folder, exist_ok=True)

        total_copied = 0
        for root, _, files in os.walk(output_base_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, output_base_folder)
                    dest_dir = project_folder if rel_path == '.' else project_folder / rel_path
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = dest_dir / file
                    shutil.copy2(source_path, dest_path)
                    total_copied += 1

        return str(project_folder), total_copied

    except Exception as e:
        print(f"[ERROR] save_frames_to_onedrive: {e}")
        return None, 0
