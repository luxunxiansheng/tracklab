from pathlib import Path

import requests
import hashlib
from tqdm import tqdm
import time
import logging


def download_file(url, local_filename, md5=None, max_retries=10, retry_delay=5):
    """
    Download file with retry logic and better error handling for proxy/network issues.
    """
    if Path(local_filename).exists():
        if md5 is not None:
            if check_md5(local_filename, md5):
                return local_filename
            else:
                print(
                    f"MD5 checksum mismatch for existing file {local_filename}, re-downloading..."
                )
                Path(local_filename).unlink()  # Remove corrupted file

    Path(local_filename).parent.mkdir(exist_ok=True, parents=True)

    for attempt in range(max_retries):
        try:
            print(
                f"Download attempt {attempt + 1}/{max_retries} for {Path(local_filename).name}"
            )

            # Use a session with longer timeout and better headers
            with requests.Session() as session:
                # session.timeout = (10, 30)  # (connect timeout, read timeout) - removed invalid assignment

                with session.get(url, stream=True, timeout=(10, 30)) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("content-length", 0))

                    # If we know the total size and file exists, try to resume
                    if total_size > 0 and Path(local_filename).exists():
                        downloaded_size = Path(local_filename).stat().st_size
                        if downloaded_size < total_size:
                            print(f"Resuming download from {downloaded_size} bytes")
                            headers = {"Range": f"bytes={downloaded_size}-"}
                            r = session.get(
                                url, stream=True, headers=headers, timeout=(10, 30)
                            )
                            r.raise_for_status()
                            mode = "ab"  # append mode
                        else:
                            print("File already complete")
                            return local_filename
                    else:
                        mode = "wb"
                        downloaded_size = 0

                    file_hash = hashlib.md5()
                    chunk_size = 65536  # Larger chunk size for better performance

                    with (
                        open(local_filename, mode) as f,
                        tqdm(
                            desc=f"Downloading {Path(local_filename).name}",
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            initial=downloaded_size,
                        ) as progress_bar,
                    ):

                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                file_hash.update(chunk)
                                progress_bar.update(len(chunk))

            # Verify download completed
            if total_size > 0:
                actual_size = Path(local_filename).stat().st_size
                if actual_size != total_size:
                    raise ValueError(
                        f"Incomplete download: expected {total_size} bytes, got {actual_size} bytes"
                    )

            # Verify MD5 if provided
            if md5 is not None:
                if md5 != file_hash.hexdigest():
                    raise ValueError(
                        f"MD5 checksum mismatch when downloading file from {url}. "
                        f"Expected: {md5}, Got: {file_hash.hexdigest()}"
                    )

            print(f"Successfully downloaded {Path(local_filename).name}")
            return local_filename

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} download attempts failed")
                raise

    return local_filename


def check_md5(local_filename, md5):
    with open(local_filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest() == md5
