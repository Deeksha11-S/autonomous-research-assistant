import json
import os
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib
from pathlib import Path


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty JSON string"""
    if isinstance(data, (dict, list)):
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    return str(data)


def validate_api_keys() -> Dict[str, bool]:
    """Validate all required API keys are present"""
    from utils.config import Config

    validation = {}

    try:
        Config.validate()
        validation["GROQ_API_KEY"] = bool(Config.GROQ_API_KEY)
    except ValueError as e:
        validation["GROQ_API_KEY"] = False

    validation["TAVILY_API_KEY"] = bool(Config.TAVILY_API_KEY)
    validation["SERPAPI_API_KEY"] = bool(Config.SERPAPI_API_KEY)

    return validation


def cleanup_temp_files(directory: str = "./data/temp",
                       max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age

    Returns:
        Number of files deleted
    """
    if not os.path.exists(directory):
        return 0

    deleted_count = 0
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
            except OSError:
                pass

        # Remove empty directories
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except OSError:
                pass

    return deleted_count


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID"""
    import uuid
    import random
    import string

    if prefix:
        prefix = prefix + "_"

    # Use UUID for uniqueness
    uid = str(uuid.uuid4()).replace("-", "")[:length]

    # Add timestamp for ordering
    timestamp = datetime.now().strftime("%H%M%S")

    return f"{prefix}{timestamp}_{uid}"


def calculate_md5(file_path: str) -> Optional[str]:
    """Calculate MD5 hash of a file"""
    if not os.path.exists(file_path):
        return None

    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except IOError:
        return None


def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if it doesn't"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[str]:
    """Get human-readable file size"""
    if not os.path.exists(file_path):
        return None

    size_bytes = os.path.getsize(file_path)

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for all OS"""
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255 - len(ext)] + ext

    return sanitized


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse various date string formats"""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(nested_dict: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten a nested dictionary"""
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)