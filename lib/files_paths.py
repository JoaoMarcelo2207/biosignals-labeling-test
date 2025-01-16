import os

# Files names
VD_INFO = "VD_INFO.CSV"
VD_FEATURES_L1 = "VD_FEATURES_L1.CSV"
VD_FEATURES_L2 = "VD_FEATURES_L2.CSV"
VD_FEATURES_L3 = "VD_FEATURES_L3.CSV"
VD_MEASURE_L0 = "VD_MEASURE_L0.CSV"
VD_LABELED_L0 = "VD_LABELED_L0.CSV"

# Dataset directories
DATASET = os.path.join("..", "Dataset")
DATASET_LOCAL = os.path.join("..", "Dataset", "DD-Local")
DATASET_YT = os.path.join("..", "Dataset", "YT-Online")
DATASET_SEED = os.path.join("..", "Dataset", "REF-Gold-Label")


# Dataset not Normalized

DATASET_NOT_NORMALIZED = os.path.join("..", "Dataset_not_normalized")
DATASET_LOCAL_NOT_NORMALIZED = os.path.join("..", "Dataset_not_normalized", "DD-Local_not_normalized")
DATASET_YT_NOT_NORMALIZED = os.path.join("..", "Dataset_not_normalized", "YT-Online_not_normalized")
DATASET_SEED_NOT_NORMALIZED = os.path.join("..", "Dataset_not_normalized", "REF-Gold-Label_not_normalized")


# Video source directories
VIDEO_SOURCE = os.path.join("..", "Video-Source")
VIDEO_SOURCE_LOCAL = os.path.join("..", "Video-Source", "in_DD-Local")
VIDEO_SOURCE_YT = os.path.join("..", "Video-Source", "in_YT-List")
VIDEO_SOURCE_SEED = os.path.join("..", "Video-Source", "in_REF-Gold")
