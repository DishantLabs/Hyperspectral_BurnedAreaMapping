"""Fetch a scene from Planet's Tanager Open Data STAC (CC-licensed).

Endpoints move, so verify the catalog URL from Planet's open-data page first.
If auto-download fails, browse the STAC in a browser, grab a basic_radiance_hdf5
asset, and drop it into data/scene.h5.
"""
import os, sys, requests

STAC = "https://www.planet.com/data/stac/open-data/tanager/catalog.json"  # verify!

def main(out="data/scene.h5"):
    try:
        from pystac_client import Client
        cat = Client.open(STAC)
        item = next(cat.get_all_items())
        asset = next(a for k, a in item.assets.items() if "radiance" in k.lower())
        print("downloading", asset.href)
        r = requests.get(asset.href, stream=True, timeout=120); r.raise_for_status()
        os.makedirs("data", exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        print("saved ->", out)
    except Exception as ex:
        print("Auto-download failed:", ex)
        print("Manual: grab a basic_radiance_hdf5 asset from the Tanager Open "
              "Data STAC and save it as data/scene.h5")

if __name__ == "__main__":
    main(*sys.argv[1:])
