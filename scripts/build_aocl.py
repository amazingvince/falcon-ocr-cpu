#!/usr/bin/env python3
"""Reproduce an isolated AOCL-DLP library build; never install or alter the runner."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys

REVISION = "abb63d85ed7a6d559ea42b5db648e2585ac9ecb8"
TAG = "AOCL-202609W02"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configured_compilers(build):
    """Read CMake's selected toolchain, not whichever compiler happens on PATH."""
    result = {}
    for language in ["C", "CXX"]:
        files = list((build / "CMakeFiles").glob("*/CMake" + language + "Compiler.cmake"))
        if len(files) != 1:
            raise ValueError(f"Expected one CMake {language} compiler identity, found {files}")
        path = files[0]
        text = path.read_text(encoding="utf-8")
        identity = {"cmake_identity_path": str(path), "cmake_identity_sha256": sha(path)}
        for field in ["COMPILER", "COMPILER_ID", "COMPILER_VERSION", "COMPILER_FRONTEND_VARIANT", "COMPILER_TARGET"]:
            match = re.search(r"set\(CMAKE_" + language + "_" + field + r' "([^"]*)"\)', text)
            if match:
                identity[field.lower()] = match.group(1)
        compiler = pathlib.Path(identity["compiler"])
        identity["compiler_sha256"] = sha(compiler)
        result[language] = identity
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=pathlib.Path("artifacts/aocl-dlp"))
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cmake", default="cmake", help="Existing CMake executable, including a project-local absolute path")
    parser.add_argument("--git-autocrlf", choices=["true", "false", "input"], help="Command-local interpretation for an existing checkout shared across operating systems; does not rewrite files or Git configuration")
    args = parser.parse_args()
    if args.jobs < 1 or args.build.exists():
        parser.error("require positive jobs and a new build directory")
    # Python's Windows environment mapping coalesces Path/PATH. Passing it
    # explicitly avoids duplicate-key failures in MSBuild's process launcher.
    env = dict(os.environ)
    cmake_path = shutil.which(args.cmake, path=env.get("PATH"))
    if cmake_path is None:
        parser.error("CMake is unavailable; pass --cmake with an existing local installation")
    cmake = str(pathlib.Path(cmake_path).resolve())
    windows = sys.platform == "win32"
    git = ["git"] + (["-c", "core.autocrlf=" + args.git_autocrlf] if args.git_autocrlf else [])
    if windows:
        env["_CL_"] = (env.get("_CL_", "") + f" /MP{args.jobs}").strip()
    if not args.source.exists():
        subprocess.run([*git, "clone", "--depth", "1", "--branch", TAG,
                        "https://github.com/amd/aocl-dlp.git", str(args.source)], env=env, check=True)
    source = args.source.resolve()
    revision = subprocess.check_output([*git, "-C", str(source), "rev-parse", "HEAD"], text=True, env=env).strip()
    if revision != REVISION:
        raise ValueError(f"AOCL revision differs: {revision}")
    dirty = subprocess.check_output([*git, "-C", str(source), "status", "--porcelain"], text=True, env=env)
    if dirty:
        raise ValueError("AOCL source checkout must be unchanged")
    tracked = subprocess.check_output([*git, "-C", str(source), "ls-files", "-z"], env=env).decode().split("\0")
    files = [name for name in tracked if name and (source / name).is_file()]
    before = {name: sha(source / name) for name in files}
    build = args.build.resolve()
    build.mkdir(parents=True)
    generator = ["-G", "Visual Studio 17 2022", "-A", "x64"] if windows else ["-G", "Unix Makefiles"]
    configure = [cmake, "-S", str(source), "-B", str(build), *generator,
                 "-DCMAKE_BUILD_TYPE=Release", "-DDLP_THREADING_MODEL=none", "-DDLP_ENABLE_OPENMP=OFF",
                 "-DBUILD_TESTING=OFF", "-DBUILD_BENCHMARKS=OFF", "-DBUILD_EXAMPLES=OFF"]
    if not windows:
        configure.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
    compile_command = [cmake, "--build", str(build), "--config", "Release", "--target", "aocl-dlp",
                       "--parallel", str(args.jobs)]
    record = {"schema_version": 2, "source_revision": revision, "source_tag": TAG,
              "source_path": str(source), "build_path": str(build),
              "source_capture_phase": "before_configure_and_after_build",
              "git_autocrlf_command_override": args.git_autocrlf,
              "script_sha256_at_start": sha(pathlib.Path(__file__)),
              "source_file_sha256": before, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "configure_command": configure, "build_command": compile_command,
              "platform": sys.platform, "platform_description": platform.platform(),
              "cmake_path": cmake, "cmake_sha256": sha(pathlib.Path(cmake)),
              "cmake_version": subprocess.check_output([cmake, "--version"], text=True, env=env),
              "threading_model": "none", "openmp": False,
              "selected_environment": {key: env[key] for key in ["CC", "CXX", "CFLAGS", "CXXFLAGS", "CL", "_CL_"] if key in env},
              "status": "building", "scope": "Standalone library for operator experiments; not linked into falcon-ocr"}
    manifest = build / "build-provenance.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        for name, command in [("configure.log", configure), ("build.log", compile_command)]:
            with (build / name).open("w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if name == "configure.log":
                record["configured_compilers"] = configured_compilers(build)
                manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        after = {name: sha(source / name) for name in files}
        if after != before:
            raise ValueError("AOCL sources changed during the build")
        if sha(pathlib.Path(__file__)) != record["script_sha256_at_start"]:
            raise ValueError("Build script changed during execution")
        pattern = "*.dll" if windows else "libaocl-dlp.so*"
        libraries = sorted({path.resolve() for path in build.rglob(pattern) if path.is_file()})
        if not libraries:
            raise ValueError("Build succeeded but no shared library was found")
        record.update(status="complete", source_unchanged_during_build=True,
                      source_file_sha256_after_build=after,
                      library_sha256={str(path): sha(path) for path in libraries},
                      cmake_cache_sha256=sha(build / "CMakeCache.txt"))
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["log_sha256"] = {name: sha(build / name) for name in ["configure.log", "build.log"] if (build / name).is_file()}
        if (build / "compile_commands.json").is_file():
            record["compile_commands_sha256"] = sha(build / "compile_commands.json")
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "libraries": record["library_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
