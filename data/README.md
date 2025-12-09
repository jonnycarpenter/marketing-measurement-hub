# Data Directory

This directory contains synthetic marketing data used by the application. The data files are stored in Google Cloud Storage and loaded at runtime.

## Data Structure

```
data/
├── master_testing_doc.csv          # Master tracker for all experiments
├── product_inventory.csv           # Product inventory data
├── product_sku_info.csv            # Product SKU details
├── promo_calendar.csv              # Promotional calendar
├── crm_sales_cust_data/
│   ├── crm_sales_data_2025_H12026.csv  # Sales transaction data
│   ├── customer_file.csv               # Customer demographics
│   └── order_line_items.csv            # Order line item details
├── test_validation_files/          # Experiment audience files
└── test_version_history/           # Version control for experiments
```

## Cloud Storage

In production, data is loaded from:
- **GCS Bucket**: `gs://ol-measurement-hub-data/`
- **Data Loader**: `utils/data_loader.py` handles GCS integration

## Local Development

For local development, you can:
1. Use `gcloud storage cp -r gs://ol-measurement-hub-data/data/ ./data/` to sync from GCS
2. Or create synthetic data using the provided schema

## Note

Data files are not tracked in git due to size constraints. All data is synthetic and generated for demonstration purposes.
