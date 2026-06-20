"""Tests for collection dataset integration."""

from pathlib import Path

import pandas as pd

from seed_preview_cv.collection.integrate_datasets import integrate_collection


def test_integrate_collection_from_synthetic_datasets(tmp_path: Path):
    ds1 = tmp_path / "dataset_1"
    ds1.mkdir()
    pd.DataFrame(
        {
            "seed": [1, 2],
            "x": [10, 20],
            "z": [-5, 6],
        }
    ).to_csv(ds1 / "spawns.csv", index=False)
    (ds1 / "1.png").write_bytes(b"png1")
    (ds1 / "2.png").write_bytes(b"png2")

    ds2 = tmp_path / "dataset_2"
    ds2.mkdir()
    pd.DataFrame(
        {
            "seed": [3],
            "x": [0],
            "z": [0],
        }
    ).to_csv(ds2 / "spawns.csv", index=False)
    (ds2 / "3.png").write_bytes(b"png3")

    screenshots = tmp_path / "data" / "screenshots"
    spawns_csv = tmp_path / "data" / "interim" / "spawns.csv"
    index_csv = tmp_path / "data" / "interim" / "collection_index.csv"
    sources_csv = tmp_path / "data" / "interim" / "collection_sources.csv"
    manifest_json = tmp_path / "data" / "interim" / "collection_manifest.json"
    seeds_txt = tmp_path / "seeds.txt"
    seeds_txt.write_text("1\n2\n3\n4\n", encoding="utf-8")

    manifest = integrate_collection(
        dataset_dirs=[ds1, ds2],
        screenshots_dir=screenshots,
        spawns_csv=spawns_csv,
        index_csv=index_csv,
        sources_csv=sources_csv,
        manifest_json=manifest_json,
        seeds_txt=seeds_txt,
        progress=False,
    )

    assert manifest["integrated_count"] == 3
    assert manifest["seeds_txt_missing"] == [4]
    assert (screenshots / "1.png").is_file()
    spawns = pd.read_csv(spawns_csv)
    assert len(spawns) == 3
    index = pd.read_csv(index_csv)
    assert set(index.columns) == {"seed", "spawn_x", "spawn_z", "screenshot_path"}
