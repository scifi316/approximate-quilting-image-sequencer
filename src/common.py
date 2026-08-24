import os

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def list_image_files(folder):
    """Return sorted image filenames (png/jpg/jpeg) in the given folder."""
    return sorted(
        filename for filename in os.listdir(folder)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )
