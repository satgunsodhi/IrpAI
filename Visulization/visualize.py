import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

from irpai import IrpAIPipeline, IrpAIConfig, Tier


def draw_regions_from_dict(draw, regions, depth=0):
    colors = ["red", "green", "blue", "orange", "magenta", "cyan"]
    color = colors[depth % len(colors)]
    for r in regions:
        bbox = r.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3 - min(2, depth))
            
            # Draw label background and text
            role = r.get("role", "unknown")
            confidence = r.get("confidence", 1.0)
            label = f"{role} ({confidence:.2f})"
            draw.text((x1 + 2, y1 + 2), label, fill=color)
            
        children = r.get("children")
        if children:
            draw_regions_from_dict(draw, children, depth + 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize IrpAI bounding boxes on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--json", type=str, help="Path to a pre-generated ContextPack JSON file (skips running pipeline).")
    parser.add_argument("--tier", type=str, choices=["lite", "standard"], default="standard", help="Pipeline tier to use (if running).")
    parser.add_argument("--out", type=str, help="Path to save the annotated image.")
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Obtain region data
    if args.json:
        print(f"Loading regions from JSON: {args.json}")
        with open(args.json, "r", encoding="utf-8") as f:
            pack_dict = json.load(f)
            regions_data = pack_dict.get("regions", [])
    else:
        print(f"Running pipeline on {image_path} (tier={args.tier})...")
        tier_enum = Tier.STANDARD if args.tier == "standard" else Tier.LITE
        pipeline = IrpAIPipeline(IrpAIConfig(tier=tier_enum))
        pack = pipeline.run(str(image_path))
        # Convert dataclasses to dicts for unified drawing logic
        regions_data = pack.to_dict(include_bboxes=True).get("regions", [])

    # 2. Draw boxes
    print("Drawing bounding boxes...")
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    draw = ImageDraw.Draw(img)
    draw_regions_from_dict(draw, regions_data)
    
    # 3. Save output
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = image_path.parent / f"{image_path.stem}_annotated.png"
        
    out_path.parent.mkdir(exist_ok=True, parents=True)
    img.save(out_path)
    print(f"Saved annotated image to {out_path.absolute()}")


if __name__ == "__main__":
    main()

