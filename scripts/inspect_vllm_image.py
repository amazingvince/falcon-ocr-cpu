#!/usr/bin/env python3
"""Read immutable public OCI metadata; never pulls large image layers."""
import hashlib
import io
import json
import pathlib
import tarfile

import requests

IMAGE_DIGEST = "sha256:5d11a0fe592de85efef88bfa5266a9392dce501e3647c6f4c69df9b74ada9afd"


def main():
    base = "https://ghcr.io/v2/tiiuae/falcon-ocr"
    token_response = requests.get("https://ghcr.io/token", params={"scope": "repository:tiiuae/falcon-ocr:pull"}, timeout=60)
    token_response.raise_for_status()
    headers = {"Authorization": "Bearer " + token_response.json()["token"],
               "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"}
    response = requests.get(base + "/manifests/" + IMAGE_DIGEST, headers=headers, timeout=60)
    response.raise_for_status()
    assert "sha256:" + hashlib.sha256(response.content).hexdigest() == IMAGE_DIGEST
    manifest = response.json()
    config_response = requests.get(base + "/blobs/" + manifest["config"]["digest"], headers=headers, timeout=60)
    config_response.raise_for_status()
    assert "sha256:" + hashlib.sha256(config_response.content).hexdigest() == manifest["config"]["digest"]
    folder = pathlib.Path("artifacts/vllm-image-metadata")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_bytes(response.content)
    (folder / "config.json").write_bytes(config_response.content)
    config = config_response.json()
    source_root = folder / "sources"
    # Only small application-source layers, in OCI order; no links or paths
    # outside /app are extracted. Weight/runtime layers remain Podman's job.
    source_layers = [16, 17, 20, 23, 25, 28]
    for index in source_layers:
        layer = manifest["layers"][index]
        assert layer["size"] < 1_000_000
        blob = requests.get(base + "/blobs/" + layer["digest"], headers=headers, timeout=60)
        blob.raise_for_status()
        assert "sha256:" + hashlib.sha256(blob.content).hexdigest() == layer["digest"]
        with tarfile.open(fileobj=io.BytesIO(blob.content), mode="r:gz") as archive:
            for item in archive:
                path = pathlib.PurePosixPath(item.name)
                if not item.isfile() or ".." in path.parts or path.is_absolute() or not path.parts or path.parts[0] != "app":
                    continue
                destination = source_root.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(item).read())
    summary = {"image": "ghcr.io/tiiuae/falcon-ocr@" + IMAGE_DIGEST,
               "config_digest": manifest["config"]["digest"], "architecture": config.get("architecture"),
               "os": config.get("os"), "created": config.get("created"),
               "compressed_layer_bytes": sum(layer["size"] for layer in manifest["layers"]),
               "layer_count": len(manifest["layers"]), "largest_layers": sorted(manifest["layers"], key=lambda l: l["size"], reverse=True)[:6],
               "entrypoint": config.get("config", {}).get("Entrypoint"), "command": config.get("config", {}).get("Cmd"),
               "environment": config.get("config", {}).get("Env"),
               "source_layer_indices": source_layers, "metadata_only": True, "runtime_status": "not_yet_launched"}
    pathlib.Path("reference/vllm-image-inspection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
