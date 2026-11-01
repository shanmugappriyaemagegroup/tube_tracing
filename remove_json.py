"""
Delete .json files from a folder, keeping the .jpg/.jpeg images.

Usage:
    python remove_json.py "C:\path\to\folder"            # delete .json files
    python remove_json.py "C:\path\to\folder" --dry-run  # only list what would be deleted
    python remove_json.py "C:\path\to\folder" -r         # include subfolders
    python remove_json.py "C:\path\to\folder" --all      # delete everything that is NOT .jpg/.jpeg
"""
import argparse
from pathlib import Path

KEEP = {".jpg", ".jpeg"}


def main():
    parser = argparse.ArgumentParser(description="Remove JSON files, keep JPG images.")
    parser.add_argument("folder", help="Folder to clean")
    parser.add_argument("-r", "--recursive", action="store_true", help="Also clean subfolders")
    parser.add_argument("--dry-run", action="store_true", help="Show files without deleting")
    parser.add_argument("--all", action="store_true", help="Delete every file that is not .jpg/.jpeg")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    files = folder.rglob("*") if args.recursive else folder.glob("*")
    deleted = kept = 0

    for f in files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        remove = (ext not in KEEP) if args.all else (ext == ".json")
        if remove:
            print(("[dry-run] " if args.dry_run else "Deleting ") + str(f))
            if not args.dry_run:
                f.unlink()
            deleted += 1
        elif ext in KEEP:
            kept += 1

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {deleted} file(s). JPG images kept: {kept}")


if __name__ == "__main__":
    main()