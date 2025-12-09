"""
Google Cloud Storage Data Loader

Provides seamless access to data files stored in GCS.
Falls back to local files for development.

Usage:
    from utils.gcs_loader import gcs_loader
    
    # Read CSV as DataFrame
    df = gcs_loader.read_csv("data/master_testing_doc.csv")
    
    # Read text file
    content = gcs_loader.read_text("agents/agent_prompts/brena_measurement_prompt.txt")
    
    # Check if file exists
    exists = gcs_loader.exists("data/some_file.csv")
    
    # List files in a prefix
    files = gcs_loader.list_files("data/test_validation_files/")
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional, List, Union
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# GCS Configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "ol-measurement-hub-data")
GCS_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "gen-lang-client-0436975498")
USE_GCS = os.environ.get("USE_GCS", "false").lower() == "true"


class GCSDataLoader:
    """
    Unified data loader that reads from GCS in production or local files in development.
    """
    
    def __init__(self):
        self._client = None
        self._bucket = None
        self._use_gcs = USE_GCS
        self._local_base = Path(__file__).parent.parent
        
    @property
    def client(self):
        """Lazy-load GCS client only when needed."""
        if self._client is None and self._use_gcs:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=GCS_PROJECT_ID)
                logger.info(f"Connected to GCS project: {GCS_PROJECT_ID}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}. Falling back to local files.")
                self._use_gcs = False
        return self._client
    
    @property
    def bucket(self):
        """Lazy-load GCS bucket only when needed."""
        if self._bucket is None and self._use_gcs:
            try:
                self._bucket = self.client.get_bucket(GCS_BUCKET_NAME)
                logger.info(f"Connected to GCS bucket: {GCS_BUCKET_NAME}")
            except Exception as e:
                logger.warning(f"Failed to access bucket {GCS_BUCKET_NAME}: {e}. Falling back to local files.")
                self._use_gcs = False
        return self._bucket
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path separators for GCS (always use forward slashes)."""
        return str(path).replace("\\", "/")
    
    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file from GCS or local filesystem.
        
        Args:
            path: Relative path to the CSV file (e.g., "data/master_testing_doc.csv")
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            DataFrame with CSV contents
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                content = blob.download_as_bytes()
                return pd.read_csv(io.BytesIO(content), **kwargs)
            except Exception as e:
                logger.warning(f"GCS read failed for {path}: {e}. Trying local file.")
        
        # Fall back to local file
        local_path = self._local_base / path
        return pd.read_csv(local_path, **kwargs)
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """
        Read a text file from GCS or local filesystem.
        
        Args:
            path: Relative path to the text file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File contents as string
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                return blob.download_as_text(encoding=encoding)
            except Exception as e:
                logger.warning(f"GCS read failed for {path}: {e}. Trying local file.")
        
        # Fall back to local file
        local_path = self._local_base / path
        return local_path.read_text(encoding=encoding)
    
    def read_bytes(self, path: str) -> bytes:
        """
        Read a binary file from GCS or local filesystem.
        
        Args:
            path: Relative path to the file
            
        Returns:
            File contents as bytes
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                return blob.download_as_bytes()
            except Exception as e:
                logger.warning(f"GCS read failed for {path}: {e}. Trying local file.")
        
        # Fall back to local file
        local_path = self._local_base / path
        return local_path.read_bytes()
    
    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> bool:
        """
        Write a DataFrame to CSV in GCS or local filesystem.
        
        Args:
            df: DataFrame to write
            path: Relative path for the CSV file
            **kwargs: Additional arguments passed to df.to_csv
            
        Returns:
            True if successful
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, **kwargs)
                blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
                logger.info(f"Wrote CSV to GCS: {path}")
                return True
            except Exception as e:
                logger.warning(f"GCS write failed for {path}: {e}. Writing to local file.")
        
        # Fall back to local file
        local_path = self._local_base / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, **kwargs)
        logger.info(f"Wrote CSV to local: {local_path}")
        return True
    
    def write_text(self, content: str, path: str, encoding: str = "utf-8") -> bool:
        """
        Write text to a file in GCS or local filesystem.
        
        Args:
            content: Text content to write
            path: Relative path for the file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            True if successful
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                blob.upload_from_string(content.encode(encoding), content_type="text/plain")
                logger.info(f"Wrote text to GCS: {path}")
                return True
            except Exception as e:
                logger.warning(f"GCS write failed for {path}: {e}. Writing to local file.")
        
        # Fall back to local file
        local_path = self._local_base / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding=encoding)
        logger.info(f"Wrote text to local: {local_path}")
        return True
    
    def exists(self, path: str) -> bool:
        """
        Check if a file exists in GCS or local filesystem.
        
        Args:
            path: Relative path to check
            
        Returns:
            True if file exists
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                return blob.exists()
            except Exception as e:
                logger.warning(f"GCS exists check failed for {path}: {e}. Checking local.")
        
        # Fall back to local file
        local_path = self._local_base / path
        return local_path.exists()
    
    def list_files(self, prefix: str, suffix: Optional[str] = None) -> List[str]:
        """
        List files with a given prefix (folder path).
        
        Args:
            prefix: Folder path prefix (e.g., "data/test_validation_files/")
            suffix: Optional file suffix filter (e.g., ".csv")
            
        Returns:
            List of file paths
        """
        prefix = self._normalize_path(prefix)
        
        if self._use_gcs:
            try:
                blobs = self.bucket.list_blobs(prefix=prefix)
                files = [blob.name for blob in blobs if not blob.name.endswith("/")]
                if suffix:
                    files = [f for f in files if f.endswith(suffix)]
                return files
            except Exception as e:
                logger.warning(f"GCS list failed for {prefix}: {e}. Listing local files.")
        
        # Fall back to local files
        local_path = self._local_base / prefix
        if not local_path.exists():
            return []
        
        files = []
        for f in local_path.rglob("*"):
            if f.is_file():
                rel_path = str(f.relative_to(self._local_base)).replace("\\", "/")
                if suffix is None or rel_path.endswith(suffix):
                    files.append(rel_path)
        return files
    
    def get_local_path(self, path: str) -> Path:
        """
        Get the local filesystem path (for backward compatibility).
        
        Args:
            path: Relative path
            
        Returns:
            Full local Path object
        """
        return self._local_base / self._normalize_path(path)
    
    def delete_file(self, path: str) -> bool:
        """
        Delete a file from GCS or local filesystem.
        
        Args:
            path: Relative path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        path = self._normalize_path(path)
        
        if self._use_gcs:
            try:
                blob = self.bucket.blob(path)
                if blob.exists():
                    blob.delete()
                    logger.info(f"Deleted from GCS: {path}")
                    return True
                else:
                    logger.warning(f"File not found in GCS: {path}")
                    return False
            except Exception as e:
                logger.warning(f"GCS delete failed for {path}: {e}. Trying local.")
        
        # Fall back to local file
        local_path = self._local_base / path
        if local_path.exists():
            local_path.unlink()
            logger.info(f"Deleted local file: {local_path}")
            return True
        return False
    
    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all files with a given prefix (folder-like delete).
        
        Args:
            prefix: Folder path prefix (e.g., "data/test_validation_files/TEST-12345")
            
        Returns:
            Number of files deleted
        """
        prefix = self._normalize_path(prefix)
        deleted_count = 0
        
        if self._use_gcs:
            try:
                blobs = list(self.bucket.list_blobs(prefix=prefix))
                for blob in blobs:
                    blob.delete()
                    deleted_count += 1
                logger.info(f"Deleted {deleted_count} files from GCS with prefix: {prefix}")
                return deleted_count
            except Exception as e:
                logger.warning(f"GCS delete_prefix failed for {prefix}: {e}. Trying local.")
        
        # Fall back to local files
        import shutil
        local_path = self._local_base / prefix
        if local_path.exists():
            if local_path.is_dir():
                # Count files before deleting
                deleted_count = sum(1 for _ in local_path.rglob("*") if _.is_file())
                shutil.rmtree(local_path)
            else:
                local_path.unlink()
                deleted_count = 1
            logger.info(f"Deleted {deleted_count} files from local: {local_path}")
        return deleted_count
    
    @property
    def is_using_gcs(self) -> bool:
        """Check if currently using GCS storage."""
        return self._use_gcs


# Global singleton instance
gcs_loader = GCSDataLoader()
