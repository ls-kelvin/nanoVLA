"""Data registry snippet for QwenPI_v4 raw-HDF5 aloha clean EEF training.

The active registry entry is defined in:
examples/Robotwin/train_files/data_registry/data_config.py
"""

from pathlib import Path


ROBOTWIN2_HDF5_ROOT = Path(
    "/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/data/RoboTwin2.0/dataset"
)


def discover_hdf5(
    tasks: list[str] | None = None,
    embodiments: list[str] | None = None,
    domains: list[str] | None = None,
    robot_type: str = "robotwin32_eef",
) -> list[tuple[str, float, str]]:
    tasks = set(tasks) if tasks is not None else None
    embodiments = set(embodiments) if embodiments is not None else None
    domains = set(domains) if domains is not None else None

    mixture = []
    for split_dir in sorted(path for path in ROBOTWIN2_HDF5_ROOT.glob("*/*") if path.is_dir()):
        task = split_dir.parent.name
        split_name = split_dir.name
        parsed = None
        for domain in ("clean", "randomized"):
            marker = f"_{domain}_"
            if marker in split_name:
                parsed = (split_name.split(marker, 1)[0], domain)
                break
        if parsed is None:
            continue
        embodiment, domain = parsed
        if tasks is not None and task not in tasks:
            continue
        if embodiments is not None and embodiment not in embodiments:
            continue
        if domains is not None and domain not in domains:
            continue
        if not (split_dir / "data").is_dir():
            continue
        mixture.append((f"{task}/{embodiment}/{domain}", 1.0, robot_type))
    return mixture


DATASET_NAMED_MIXTURES = {
    "hdf5_aloha_clean_eef": discover_hdf5(
        embodiments=["aloha-agilex"],
        domains=["clean"],
        robot_type="robotwin32_eef",
    )
}
