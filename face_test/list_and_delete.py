#!/usr/bin/env python3
"""List local enrollment names and embedding counts, or delete an enrollment."""

import argparse
import sys

import face_core as core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", help="Delete the enrollment with this exact name")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.delete:
        try:
            core.validate_person_name(args.delete)
        except ValueError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        path = core.DATA_DIR / f"{args.delete}.npy"
        metadata_path = core.DATA_DIR / f"{args.delete}_meta.json"
        if not path.is_file():
            print(f"No enrollment found for {args.delete!r}.", file=sys.stderr)
            return 1
        try:
            path.unlink()
            if metadata_path.is_file():
                metadata_path.unlink()
        except OSError as error:
            print(f"Could not delete enrollment {path}: {error}", file=sys.stderr)
            return 1
        print(f"Deleted enrollment and metadata for {args.delete!r}.")
        return 0

    try:
        enrollments = core.load_enrollments()
    except FileNotFoundError:
        print("No enrolled people.")
        return 0
    except ValueError as error:
        print(f"Enrollment data error: {error}", file=sys.stderr)
        return 1
    print("Enrolled people:")
    for name in sorted(enrollments):
        print(f"  {name}: {len(enrollments[name])} embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
