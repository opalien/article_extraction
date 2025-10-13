from __future__ import annotations

from typing import Dict


# Mapping from ORM attribute names to CSV column names
COLUMN_MAPPING: Dict[str, str] = {
    "id_paper": "id_paper",
    "model": "Model",
    "domain": "Domain",
    "task": "Task",
    "organization": "Organization",
    "authors": "Authors",
    "publication_date": "Publication date",
    "reference": "Reference",
    "link": "Link",
    "citations": "Citations",
    "notability_criteria": "Notability criteria",
    "notability_criteria_notes": "Notability criteria notes",
    "parameters": "Parameters",
    "parameters_notes": "Parameters notes",
    "training_compute": "Training compute (FLOP)",
    "training_compute_notes": "Training compute notes",
    "training_dataset": "Training dataset",
    "training_dataset_notes": "Training dataset notes",
    "training_dataset_size_datapoints": "Training dataset size (datapoints)",
    "dataset_size_notes": "Dataset size notes",
    "training_time_hours": "Training time (hours)",
    "training_time_notes": "Training time notes",
    "training_hardware": "Training hardware",
    "approach": "Approach",
    "confidence": "Confidence",
    "abstract": "Abstract",
    "epochs": "Epochs",
    "benchmark_data": "Benchmark data",
    "model_accessibility": "Model accessibility",
    "country_of_organization": "Country (of organization)",
    "base_model": "Base model",
    "finetune_compute": "Finetune compute (FLOP)",
    "finetune_compute_notes": "Finetune compute notes",
    "hardware_quantity": "Hardware quantity",
    "hardware_utilization_mfu": "Hardware utilization (MFU)",
    "last_modified": "Last modified",
    "training_cloud_compute_vendor": "Training cloud compute vendor",
    "training_data_center": "Training data center",
    "archived_links": "Archived links",
    "batch_size": "Batch size",
    "batch_size_notes": "Batch size notes",
    "organization_categorization": "Organization categorization",
    "foundation_model": "Foundation model",
    "training_compute_lower_bound": "Training compute lower bound",
    "training_compute_upper_bound": "Training compute upper bound",
    "training_chip_hours": "Training chip-hours",
    "training_code_accessibility": "Training code accessibility",
    "accessibility_notes": "Accessibility notes",
    "organization_categorization_from_organization": "Organization categorization (from Organization)",
    "possibly_over_1e23_flop": "Possibly over 1e23 FLOP",
    "training_compute_cost_2023_usd": "Training compute cost (2023 USD)",
    "utilization_notes": "Utilization notes",
    "numerical_format": "Numerical format",
    "frontier_model": "Frontier model",
    "training_power_draw_w": "Training power draw (W)",
    "training_compute_estimation_method": "Training compute estimation method",
    "hugging_face_developer_id": "Hugging Face developer id",
    "post_training_compute_flop": "Post-training compute (FLOP)",
    "post_training_compute_notes": "Post-training compute notes",
    "hardware_utilization_hfu": "Hardware utilization (HFU)",
}

COLUMN_ORDER = list(COLUMN_MAPPING.keys())

COLUMN_CSV_TO_ATTR = {csv: attr for attr, csv in COLUMN_MAPPING.items() if csv}

DATE_COLUMNS = {"publication_date"}
DATETIME_COLUMNS = {"last_modified"}
INTEGER_COLUMNS = {"id_paper", "citations"}
FLOAT_COLUMNS = {
    "parameters",
    "training_compute",
    "training_dataset_size_datapoints",
    "training_time_hours",
    "epochs",
    "finetune_compute",
    "hardware_quantity",
    "hardware_utilization_mfu",
    "batch_size",
    "training_compute_lower_bound",
    "training_compute_upper_bound",
    "training_chip_hours",
    "training_compute_cost_2023_usd",
    "training_power_draw_w",
    "post_training_compute_flop",
    "hardware_utilization_hfu",
}
BOOLEAN_COLUMNS = {"possibly_over_1e23_flop", "frontier_model"}


