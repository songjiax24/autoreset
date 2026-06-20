"""Project path constants."""

from pathlib import Path

# src/seed_preview_cv/common/paths.py -> project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

CUBIOMES_DIR = THIRD_PARTY_DIR / "cubiomes"

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_SCREENSHOTS_DIR = DATA_DIR / "screenshots"
DATA_SCREENSHOTS_MASKED_DIR = DATA_DIR / "screenshots_masked"
DATA_DATASET_IMAGES_DIR = DATA_DIR / "dataset_images"
DATA_SCREENSHOTS_ANNOTATED_DIR = DATA_DIR / "screenshots_annotated"
DATA_SCREENSHOTS_ANNOTATED_DIRECTIONAL_DIR = DATA_DIR / "screenshots_annotated_directional"
DATA_SCREENSHOTS_ANNOTATED_COMPARE_DIR = DATA_DIR / "screenshots_annotated_compare"
DATA_LABELS_DIR = DATA_DIR / "labels"

# 测试集：未经筛选的随机种子（5000），勿与平衡主集 28000 混用
DATASET_TEST_RANDOM_DIR = PROJECT_ROOT / "dataset_test_random"
TEST_RANDOM_SPAWNS_CSV = DATA_INTERIM_DIR / "test_random_spawns.csv"
TEST_RANDOM_DATASET_IMAGES_DIR = DATA_DATASET_IMAGES_DIR / "test_random"
TEST_RANDOM_LABELS_ISOTROPIC_CSV = DATA_LABELS_DIR / "test_random_labels_isotropic.csv"
TEST_RANDOM_LABELS_DIRECTIONAL_CSV = DATA_LABELS_DIR / "test_random_labels_directional.csv"

SEEDS_TXT = PROJECT_ROOT / "seeds.txt"
SEEDS_META_JSON = PROJECT_ROOT / "seeds.meta.json"

# 环节 2：mod 采集结果
COLLECTION_SPAWNS_CSV = DATA_INTERIM_DIR / "spawns.csv"
COLLECTION_INDEX_CSV = DATA_INTERIM_DIR / "collection_index.csv"
COLLECTION_SOURCES_CSV = DATA_INTERIM_DIR / "collection_sources.csv"
COLLECTION_MANIFEST_JSON = DATA_INTERIM_DIR / "collection_manifest.json"
