"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


DEFAULT_SOURCE_DIRS = {
    "python": ("python", "triton"),
    "triton": ("triton",),
    "cuda": ("cuda",),
    "cpp": ("cpp",),
    "tilelang": ("tilelang",),
}

DEFAULT_ENTRY_FILES = {
    "python": ("kernel.py", "main.py"),
    "triton": ("kernel.py", "main.py"),
    "cuda": ("kernel.cu", "main.cu", "kernel.cpp", "main.cpp"),
    "cpp": ("kernel.cpp", "main.cpp", "kernel.cc", "main.cc"),
    "tilelang": ("kernel.py", "main.py"),
}

SOURCE_FILE_SUFFIXES = {
    "python": (".py",),
    "triton": (".py",),
    "cuda": (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".py"),
    "cpp": (".cpp", ".cc", ".c", ".h", ".hpp"),
    "tilelang": (".py",),
}

TRACK_ALIASES = {
    "fused_moe": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    "sparse_attention": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "gated_delta_net": "gdn_decode_qk4_v8_d128_k_last",
}


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def validate_definition(definition: str) -> None:
    """Reject track aliases early with an actionable message."""
    if definition in TRACK_ALIASES:
        raise ValueError(
            f"Invalid definition '{definition}'. Use a concrete dataset definition name "
            f"in config.toml instead, for example '{TRACK_ALIASES[definition]}'."
        )


def _find_source_files(source_dir: Path, language: str) -> list[Path]:
    suffixes = SOURCE_FILE_SUFFIXES[language]
    return sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix in suffixes
    )


def resolve_source_dir(build_config: dict, language: str) -> Path:
    """Resolve the source directory from config or infer it from the language."""
    explicit_source_dir = build_config.get("source_dir")
    solution_root = PROJECT_ROOT / "solution"

    if explicit_source_dir:
        source_dir = solution_root / explicit_source_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Configured source_dir not found: {source_dir}")
        return source_dir

    for relative_dir in DEFAULT_SOURCE_DIRS[language]:
        candidate = solution_root / relative_dir
        if candidate.exists() and _find_source_files(candidate, language):
            return candidate

    checked_dirs = ", ".join(str(solution_root / item) for item in DEFAULT_SOURCE_DIRS[language])
    raise FileNotFoundError(
        f"No source directory found for language '{language}'. Checked: {checked_dirs}. "
        "You can also set [build].source_dir in config.toml."
    )


def _guess_entry_file(source_dir: Path, language: str) -> str:
    for filename in DEFAULT_ENTRY_FILES[language]:
        if (source_dir / filename).exists():
            return filename

    source_files = _find_source_files(source_dir, language)
    if source_files:
        return source_files[0].name

    raise FileNotFoundError(f"No source files found in {source_dir}")


def normalize_entry_point(entry_point: str, language: str, source_dir: Path) -> str:
    """Normalize legacy function-only entry points to the current file::function format."""
    if "::" in entry_point:
        return entry_point

    if "/" in entry_point or "\\" in entry_point:
        raise ValueError(
            f"Invalid entry point '{entry_point}'. Expected '<file_path>::<function_name>'."
        )

    entry_file = _guess_entry_file(source_dir, language)
    return f"{entry_file}::{entry_point}"


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]
    validate_definition(solution_config["definition"])

    language = build_config["language"]
    if language not in DEFAULT_SOURCE_DIRS:
        raise ValueError(f"Unsupported language: {language}")

    source_dir = resolve_source_dir(build_config, language)
    entry_point = normalize_entry_point(build_config["entry_point"], language, source_dir)

    # Create build spec
    dps = build_config.get("destination_passing_style", True)
    dependencies = build_config.get("dependencies", [])
    binding = build_config.get("binding")
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        dependencies=dependencies,
        destination_passing_style=dps,
        binding=binding,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")
    print(f"  Source dir: {source_dir.relative_to(PROJECT_ROOT)}")
    print(f"  Entry point: {entry_point}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
