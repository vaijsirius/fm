"""Reusable S3 reader for the batch3 10k dataset.

Usage as a module:
    from s3_reader import S3Reader
    reader = S3Reader()                       # uses default AWS credentials/profile
    keys = reader.list_files()                # all object keys under the prefix
    df = reader.read_file(keys[0])            # one file -> pandas DataFrame
    df_all = reader.read_all()                # every file concatenated

Usage from the command line:
    python s3_reader.py                       # list files
    python s3_reader.py --preview             # list + preview the first file
    python s3_reader.py --profile myprofile --out combined.csv

Requires: pip install boto3 pandas  (pyarrow for parquet, openpyxl for xlsx)
"""
import argparse
import io
import json
import os
from typing import Iterator, List, Optional
from urllib.parse import urlparse

import boto3
import pandas as pd

DEFAULT_S3_URI = (
    "s3://ld-paypalb1-data-448049797912/Learning and Development/"
    "Batches/batch3_data_10k/10k_data/"
)


class S3Reader:
    def __init__(self, s3_uri: str = DEFAULT_S3_URI,
                 profile: Optional[str] = None,
                 region: Optional[str] = None):
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Not an S3 URI: {s3_uri}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.lstrip("/")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"
        session = boto3.Session(profile_name=profile, region_name=region)
        self.s3 = session.client("s3")

    # ---------- listing ----------
    def list_files(self, extensions: Optional[tuple] = None) -> List[str]:
        """Return object keys under the prefix (folders skipped)."""
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if extensions and not key.lower().endswith(extensions):
                    continue
                keys.append(key)
        return keys

    # ---------- reading ----------
    def read_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def read_file(self, key: str, **kwargs) -> pd.DataFrame:
        """Read one object into a DataFrame, chosen by file extension.

        Extra kwargs are passed to the underlying pandas reader.
        """
        name = key.lower()
        data = io.BytesIO(self.read_bytes(key))
        if name.endswith((".csv", ".csv.gz")):
            return pd.read_csv(data, **kwargs)
        if name.endswith((".tsv", ".tsv.gz")):
            return pd.read_csv(data, sep="\t", **kwargs)
        if name.endswith((".jsonl", ".ndjson")):
            return pd.read_json(data, lines=True, **kwargs)
        if name.endswith(".json"):
            try:
                return pd.read_json(data, **kwargs)
            except ValueError:
                data.seek(0)
                return pd.json_normalize(json.load(data))
        if name.endswith(".parquet"):
            return pd.read_parquet(data, **kwargs)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(data, **kwargs)
        if name.endswith((".txt", ".log")):
            text = data.read().decode("utf-8", errors="replace")
            return pd.DataFrame({"text": text.splitlines()})
        raise ValueError(f"Unsupported file type: {key}")

    def iter_files(self, extensions: Optional[tuple] = None,
                   **kwargs) -> Iterator[tuple]:
        """Yield (key, DataFrame) one file at a time (memory friendly)."""
        for key in self.list_files(extensions):
            try:
                yield key, self.read_file(key, **kwargs)
            except ValueError as e:
                print(f"Skipping {key}: {e}")

    def read_all(self, extensions: Optional[tuple] = None,
                 add_source: bool = True, **kwargs) -> pd.DataFrame:
        """Read every supported file and concatenate into one DataFrame."""
        frames = []
        for key, df in self.iter_files(extensions, **kwargs):
            if add_source:
                df["source_file"] = os.path.basename(key)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def download_all(self, local_dir: str) -> List[str]:
        """Download raw files to a local folder; returns local paths."""
        os.makedirs(local_dir, exist_ok=True)
        paths = []
        for key in self.list_files():
            dest = os.path.join(local_dir, os.path.basename(key))
            self.s3.download_file(self.bucket, key, dest)
            paths.append(dest)
        return paths


def main():
    ap = argparse.ArgumentParser(description="Read files from S3 prefix")
    ap.add_argument("--uri", default=DEFAULT_S3_URI)
    ap.add_argument("--profile", default=None, help="AWS profile name")
    ap.add_argument("--region", default=None)
    ap.add_argument("--preview", action="store_true",
                    help="Show head of the first file")
    ap.add_argument("--out", help="Write all files combined to this CSV")
    args = ap.parse_args()

    reader = S3Reader(args.uri, args.profile, args.region)
    keys = reader.list_files()
    print(f"Found {len(keys)} files under s3://{reader.bucket}/{reader.prefix}")
    for k in keys[:20]:
        print("  ", k)
    if len(keys) > 20:
        print(f"   ... and {len(keys) - 20} more")

    if args.preview and keys:
        df = reader.read_file(keys[0])
        print(f"\nPreview of {keys[0]} (shape={df.shape}):")
        print(df.head())

    if args.out:
        df = reader.read_all()
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
