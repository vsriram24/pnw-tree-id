"""Download tree images from iNaturalist using the pyinaturalist API."""

import hashlib
import os
import re
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


def load_species_config(config_path: str = "config/species.yaml") -> dict:
    """Load species configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def species_slug(common_name: str) -> str:
    """Convert common name to directory-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", common_name.lower()).strip("_")


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image from a URL."""
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except (requests.RequestException, OSError):
        return False


def fetch_observations(
    taxon_id: int,
    place_ids: list[int],
    quality_grade: str = "research",
    per_page: int = 200,
    max_images: int = 400,
) -> list[dict]:
    """Fetch observation photo URLs from iNaturalist API.

    Uses the v1 observations API directly to avoid pyinaturalist version issues.
    """
    photos = []
    page = 1
    seen_ids = set()

    while len(photos) < max_images:
        params = {
            "taxon_id": taxon_id,
            "place_id": ",".join(str(p) for p in place_ids),
            "quality_grade": quality_grade,
            "photos": "true",
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "votes",
        }

        try:
            resp = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  API error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            obs_id = obs["id"]
            if obs_id in seen_ids:
                continue
            seen_ids.add(obs_id)

            for idx, photo in enumerate(obs.get("photos", [])):
                if len(photos) >= max_images:
                    break
                url = photo.get("url", "")
                if url:
                    # Replace square thumbnail with medium-sized image
                    url = url.replace("/square.", "/medium.")
                    photos.append({
                        "obs_id": obs_id,
                        "photo_idx": idx,
                        "url": url,
                    })

        page += 1
        if page > data.get("total_results", 0) // per_page + 1:
            break

    return photos[:max_images]


def download_species_images(
    species_entry: dict,
    place_ids: list[int],
    output_dir: str = "data/raw",
    max_images: int = 400,
    quality_grade: str = "research",
    rate_limit: float = 1.0,
) -> int:
    """Download images for a single species.

    Returns the number of images successfully downloaded.
    """
    common = species_entry["common_name"]
    taxon_id = species_entry["taxon_id"]
    slug = species_slug(common)
    species_dir = Path(output_dir) / slug
    species_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = set(p.name for p in species_dir.glob("*.jpg"))
    if len(existing) >= max_images:
        print(f"  {common}: already have {len(existing)} images, skipping")
        return len(existing)

    print(f"  Fetching observations for {common} (taxon {taxon_id})...")
    photos = fetch_observations(
        taxon_id=taxon_id,
        place_ids=place_ids,
        quality_grade=quality_grade,
        max_images=max_images,
    )

    downloaded = 0
    for photo in tqdm(photos, desc=f"  {common}", unit="img"):
        filename = f"{photo['obs_id']}_{photo['photo_idx']}.jpg"
        if filename in existing:
            downloaded += 1
            continue

        save_path = species_dir / filename
        if download_image(photo["url"], save_path):
            downloaded += 1
        time.sleep(rate_limit)

    return downloaded


def download_all(
    config_path: str = "config/species.yaml",
    output_dir: str = "data/raw",
    species_filter: list[str] | None = None,
) -> dict[str, int]:
    """Download images for all species in the config.

    Args:
        config_path: Path to species YAML config.
        output_dir: Directory to save images.
        species_filter: Optional list of common names to download (subset).

    Returns:
        Dict mapping species common name to count of downloaded images.
    """
    config = load_species_config(config_path)
    place_ids = list(config["place_ids"].values())
    dl_config = config["download"]
    max_images = dl_config["images_per_species"]
    quality_grade = dl_config["quality_grade"]
    rate_limit = dl_config["rate_limit_seconds"]

    results = {}
    species_list = config["species"]
    if species_filter:
        filter_lower = {s.lower() for s in species_filter}
        species_list = [
            s for s in species_list if s["common_name"].lower() in filter_lower
        ]

    print(f"Downloading images for {len(species_list)} species...")
    for entry in species_list:
        count = download_species_images(
            species_entry=entry,
            place_ids=place_ids,
            output_dir=output_dir,
            max_images=max_images,
            quality_grade=quality_grade,
            rate_limit=rate_limit,
        )
        results[entry["common_name"]] = count
        print(f"  {entry['common_name']}: {count} images")

    return results
