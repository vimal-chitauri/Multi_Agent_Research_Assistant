import os
from dotenv import load_dotenv

load_dotenv(override=True)


def _configure():
    import cloudinary
    cloudinary.config(
        cloud_name  = os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key     = os.getenv("CLOUDINARY_API_KEY"),
        api_secret  = os.getenv("CLOUDINARY_API_SECRET"),
        secure      = True,
    )


def upload_to_cloudinary(video_path: str) -> tuple[str, str, str] | tuple[None, None, None]:
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key    = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if not all([cloud_name, api_key, api_secret]):
        print("  Cloudinary: CLOUDINARY_CLOUD_NAME / API_KEY / API_SECRET not set in .env")
        return None, None, None

    try:
        import cloudinary.uploader
        _configure()

        file_name = os.path.basename(video_path)
        public_id = f"marketing_agent/{file_name.replace('.mp4', '')}"

        print(f"  Cloudinary: uploading {file_name}...")
        result = cloudinary.uploader.upload(
            video_path,
            resource_type = "video",
            public_id     = public_id,
        )

        url  = result["secure_url"]
        size_mb = os.path.getsize(video_path) / 1_048_576
        print(f"  Cloudinary: uploaded ({size_mb:.1f} MB) → {url}")
        return url, public_id, file_name

    except Exception as e:
        print(f"  Cloudinary upload error: {e}")
        return None, None, None


def delete_from_cloudinary(public_id: str, file_name: str = "") -> bool:
    try:
        import cloudinary.uploader
        _configure()

        result = cloudinary.uploader.destroy(public_id, resource_type="video")
        if result.get("result") == "ok":
            print(f"  Cloudinary: deleted {public_id}")
            return True
        print(f"  Cloudinary: delete result — {result}")
    except Exception as e:
        print(f"  Cloudinary delete error: {e}")
    return False


def upload_to_b2(video_path: str) -> tuple[str, str, str] | tuple[None, None, None]:
    return upload_to_cloudinary(video_path)


def delete_from_b2(file_id: str, file_name: str) -> bool:
    return delete_from_cloudinary(file_id, file_name)
