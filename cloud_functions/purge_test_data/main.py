"""
Cloud Function to purge test data daily.
Resets master_testing_doc.csv to baseline and removes all test artifacts
except for the two baseline tests (TEST-12022008, TEST-12022131).
"""

from google.cloud import storage
import functions_framework


BUCKET_NAME = "ol-measurement-hub-data"
BASELINE_TESTS = {"TEST-12022008", "TEST-12022131"}


@functions_framework.http
def purge_test_data(request):
    """HTTP Cloud Function to purge test data."""
    
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    deleted_files = []
    kept_files = []
    
    # 1. Reset master_testing_doc.csv from baseline
    baseline_blob = bucket.blob("data/master_testing_doc_baseline.csv")
    master_blob = bucket.blob("data/master_testing_doc.csv")
    
    # Copy baseline to master
    bucket.copy_blob(baseline_blob, bucket, "data/master_testing_doc.csv")
    kept_files.append("data/master_testing_doc.csv (reset from baseline)")
    
    # 2. Purge test_audience_files (except baseline tests)
    for blob in bucket.list_blobs(prefix="data/test_audience_files/"):
        if blob.name.endswith("/"):
            continue
        test_id = extract_test_id(blob.name)
        if test_id and test_id not in BASELINE_TESTS:
            blob.delete()
            deleted_files.append(blob.name)
        else:
            kept_files.append(blob.name)
    
    # 3. Purge test_validation_files (except baseline tests)
    for blob in bucket.list_blobs(prefix="data/test_validation_files/"):
        if blob.name.endswith("/"):
            continue
        test_id = extract_test_id(blob.name)
        if test_id and test_id not in BASELINE_TESTS:
            blob.delete()
            deleted_files.append(blob.name)
        else:
            kept_files.append(blob.name)
    
    # 4. Purge test_version_history (except baseline tests)
    for blob in bucket.list_blobs(prefix="data/test_version_history/"):
        if blob.name.endswith("/"):
            continue
        test_id = extract_test_id(blob.name)
        if test_id and test_id not in BASELINE_TESTS:
            blob.delete()
            deleted_files.append(blob.name)
        else:
            kept_files.append(blob.name)
    
    # 5. Purge session_artifacts (delete all - these are ephemeral)
    for blob in bucket.list_blobs(prefix="data/session_artifacts/"):
        if blob.name.endswith("/"):
            continue
        blob.delete()
        deleted_files.append(blob.name)
    
    return {
        "status": "success",
        "deleted_count": len(deleted_files),
        "kept_count": len(kept_files),
        "deleted_files": deleted_files,
        "kept_files": kept_files
    }


def extract_test_id(filename: str) -> str:
    """Extract TEST-XXXXXXXX from a filename."""
    import re
    match = re.search(r'(TEST-\d{8})', filename)
    return match.group(1) if match else None
