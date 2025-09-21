#!/usr/bin/env python3
"""
YOLOv12-DINO Inference Script

This script performs object detection on images using trained YOLOv12-DINO model weights.
It can process single images, image directories, or image lists and generate annotated outputs.

Usage:
    python inference.py --weights path/to/model.pt --source path/to/images --output path/to/output
    python inference.py --weights runs/detect/train/weights/best.pt --source test_images/ --output results/
    python inference.py --weights best.pt --source image.jpg --conf 0.5 --iou 0.7 --save --show

Features:
    - Supports multiple input formats (single image, directory, image list)
    - Configurable confidence and IoU thresholds
    - Optional visualization and saving of annotated images
    - Batch processing for efficient inference
    - Support for various image formats (jpg, png, bmp, etc.)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Union
import time

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.utils.checks import check_file, check_imgsz
    from ultralytics.utils.files import increment_path
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    print("Please ensure ultralytics is properly installed")
    sys.exit(1)


class YOLOInference:
    """YOLOv12-DINO inference class for object detection on images."""

    def __init__(
        self,
        weights: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str = "",
        verbose: bool = True
    ):
        """
        Initialize the YOLOv12-DINO inference model.

        Args:
            weights (str | Path): Path to the trained model weights (.pt file)
            conf (float): Confidence threshold for detection (0.0-1.0)
            iou (float): IoU threshold for Non-Maximum Suppression (0.0-1.0)
            imgsz (int): Input image size for inference
            device (str): Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            verbose (bool): Enable verbose output
        """
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = check_imgsz(imgsz)
        self.device = device
        self.verbose = verbose

        # Validate weights file
        if not self.weights.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights}")

        # Load model
        if self.verbose:
            LOGGER.info(f"Loading YOLOv12-DINO model from {self.weights}")
        
        self.model = YOLO(str(self.weights), verbose=self.verbose)
        
        if self.verbose:
            LOGGER.info(f"Model loaded successfully")
            LOGGER.info(f"Model task: {self.model.task}")
            if hasattr(self.model.model, 'names'):
                LOGGER.info(f"Classes: {list(self.model.model.names.values())}")

    def predict_single(
        self,
        source: Union[str, Path],
        save: bool = False,
        show: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        output_dir: Union[str, Path] = None
    ):
        """
        Perform inference on a single image source.

        Args:
            source (str | Path): Path to image file
            save (bool): Save annotated images
            show (bool): Display results
            save_txt (bool): Save detection results to txt files
            save_conf (bool): Save confidence scores in txt files
            save_crop (bool): Save cropped detection images
            output_dir (str | Path): Output directory for saved results

        Returns:
            List of Results objects containing detection results
        """
        # Prepare prediction arguments
        predict_args = {
            'source': str(source),
            'conf': self.conf,
            'iou': self.iou,
            'imgsz': self.imgsz,
            'save': save,
            'show': show,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'save_crop': save_crop,
            'verbose': self.verbose
        }

        if self.device:
            predict_args['device'] = self.device

        if output_dir:
            predict_args['project'] = str(Path(output_dir).parent)
            predict_args['name'] = Path(output_dir).name

        # Run inference
        results = self.model.predict(**predict_args)
        return results

    def predict_batch(
        self,
        source_dir: Union[str, Path],
        save: bool = True,
        show: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        output_dir: Union[str, Path] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    ):
        """
        Perform batch inference on all images in a directory.

        Args:
            source_dir (str | Path): Directory containing images
            save (bool): Save annotated images
            show (bool): Display results
            save_txt (bool): Save detection results to txt files
            save_conf (bool): Save confidence scores in txt files
            save_crop (bool): Save cropped detection images
            output_dir (str | Path): Output directory for saved results
            extensions (tuple): Supported image file extensions

        Returns:
            List of Results objects containing detection results for all images
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(source_dir.glob(f"*{ext}"))
            image_files.extend(source_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            raise ValueError(f"No images found in {source_dir} with extensions {extensions}")

        if self.verbose:
            LOGGER.info(f"Found {len(image_files)} images in {source_dir}")

        # Prepare prediction arguments
        predict_args = {
            'source': str(source_dir),
            'conf': self.conf,
            'iou': self.iou,
            'imgsz': self.imgsz,
            'save': save,
            'show': show,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'save_crop': save_crop,
            'verbose': self.verbose
        }

        if self.device:
            predict_args['device'] = self.device

        if output_dir:
            predict_args['project'] = str(Path(output_dir).parent)
            predict_args['name'] = Path(output_dir).name

        # Run batch inference
        results = self.model.predict(**predict_args)
        return results

    def predict_from_list(
        self,
        image_list: List[Union[str, Path]],
        save: bool = True,
        show: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        output_dir: Union[str, Path] = None
    ):
        """
        Perform inference on a list of image paths.

        Args:
            image_list (List[str | Path]): List of image file paths
            save (bool): Save annotated images
            show (bool): Display results
            save_txt (bool): Save detection results to txt files
            save_conf (bool): Save confidence scores in txt files
            save_crop (bool): Save cropped detection images
            output_dir (str | Path): Output directory for saved results

        Returns:
            List of Results objects containing detection results for all images
        """
        all_results = []

        for image_path in image_list:
            if not Path(image_path).exists():
                LOGGER.warning(f"Image not found: {image_path}")
                continue

            results = self.predict_single(
                source=image_path,
                save=save,
                show=show,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                output_dir=output_dir
            )
            all_results.extend(results)

        return all_results

    def print_results_summary(self, results, source_info: str = ""):
        """Print a summary of detection results."""
        if not results:
            LOGGER.info(f"No results to display{f' for {source_info}' if source_info else ''}")
            return

        total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
        
        if self.verbose:
            LOGGER.info(f"\n{colorstr('Results Summary')}{f' for {source_info}' if source_info else ''}:")
            LOGGER.info(f"  Images processed: {len(results)}")
            LOGGER.info(f"  Total detections: {total_detections}")

            if hasattr(self.model.model, 'names') and total_detections > 0:
                # Count detections per class
                class_counts = {}
                for result in results:
                    if result.boxes is not None:
                        for cls in result.boxes.cls:
                            cls_name = self.model.model.names[int(cls)]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                LOGGER.info("  Detections by class:")
                for cls_name, count in sorted(class_counts.items()):
                    LOGGER.info(f"    {cls_name}: {count}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv12-DINO Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image inference
    python inference.py --weights best.pt --source image.jpg --save --show

    # Batch inference on directory
    python inference.py --weights runs/detect/train/weights/best.pt --source test_images/ --output results/

    # Custom thresholds and save options
    python inference.py --weights model.pt --source images/ --conf 0.5 --iou 0.7 --save-txt --save-crop
        """
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source for inference (image file, directory, or list of images)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: runs/detect/predict)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detection (default: 0.25)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold for NMS (default: 0.7)'
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to run on (cpu, cuda, mps, etc.) (default: auto-detect)'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save annotated images'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results'
    )

    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save detection results to txt files'
    )

    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidence scores in txt files'
    )

    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='Save cropped detection images'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()

    try:
        # Initialize inference model
        inference = YOLOInference(
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=args.verbose
        )

        # Determine source type and run inference
        source_path = Path(args.source)
        start_time = time.time()

        if source_path.is_file():
            # Single image inference
            if args.verbose:
                LOGGER.info(f"Running inference on single image: {source_path}")
            
            results = inference.predict_single(
                source=source_path,
                save=args.save,
                show=args.show,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_crop=args.save_crop,
                output_dir=args.output
            )
            
            inference.print_results_summary(results, str(source_path))

        elif source_path.is_dir():
            # Directory batch inference
            if args.verbose:
                LOGGER.info(f"Running batch inference on directory: {source_path}")
            
            results = inference.predict_batch(
                source_dir=source_path,
                save=args.save,
                show=args.show,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_crop=args.save_crop,
                output_dir=args.output
            )
            
            inference.print_results_summary(results, str(source_path))

        else:
            raise FileNotFoundError(f"Source not found: {source_path}")

        # Print timing information
        end_time = time.time()
        if args.verbose:
            LOGGER.info(f"\nInference completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        LOGGER.error(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()