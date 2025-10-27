import argparse
import os

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="strakammm/generals_io_replays")
    parser.add_argument(
        "--local-dir",
        default=os.path.join(os.path.dirname(__file__), "generals_io_replays"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    path = snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=args.hf_token,
    )
    print(path)


if __name__ == "__main__":
    main()

