#!/usr/bin/env python3
"""
YOLOv12-DINO Segmentation Inference Script

This script performs instance segmentation on images using trained YOLOv12-DINO segmentation model weights.
It can process single images, image directories, or image lists and generate annotated outputs with precise masks.

Usage:
    python inference.py --weights path/to/model.pt --source path/to/images --output path/to/output
    python inference.py --weights runs/segment/train/weights/best.pt --source test_images/ --output results/
    python inference.py --weights best.pt --source image.jpg --conf 0.5 --iou 0.7 --save --show

Features:
    - Instance segmentation with pixel-perfect masks
    - Supports multiple input formats (single image, directory, image list)
    - Configurable confidence and IoU thresholds
    - Optional visualization and saving of annotated images with masks
    - Batch processing for efficient segmentation inference
    - Support for various image formats (jpg, png, bmp, etc.)
    - Mask visualization and export options
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


class YOLOSegmentationInference:
    """YOLOv12-DINO inference class for instance segmentation on images."""

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
        Initialize the YOLOv12-DINO segmentation model.

        Args:
            weights (str | Path): Path to the trained segmentation model weights (.pt file)
            conf (float): Confidence threshold for instance detection (0.0-1.0)
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
            LOGGER.info(f"Loading YOLOv12-DINO segmentation model from {self.weights}")
        
        self.model = YOLO(str(self.weights), verbose=self.verbose)
        
        if self.verbose:
            LOGGER.info(f"Segmentation model loaded successfully")
            LOGGER.info(f"Model task: {self.model.task}")
            if hasattr(self.model.model, 'names'):
                LOGGER.info(f"Classes: {list(self.model.model.names.values())}")
                
        # Verify this is a segmentation model
        if hasattr(self.model, 'task') and self.model.task != 'segment':
            LOGGER.warning(f"Model task is '{self.model.task}' but expected 'segment'. Results may be unexpected.")

    def predict_single(
        self,
        source: Union[str, Path],
        save: bool = False,
        show: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        save_masks: bool = False,
        output_dir: Union[str, Path] = None
    ):
        """
        Perform segmentation inference on a single image source.

        Args:
            source (str | Path): Path to image file
            save (bool): Save annotated images with masks
            show (bool): Display segmentation results
            save_txt (bool): Save segmentation results to txt files
            save_conf (bool): Save confidence scores in txt files
            save_crop (bool): Save cropped instance images
            save_masks (bool): Save instance masks as separate images
            output_dir (str | Path): Output directory for saved results

        Returns:
            List of Results objects containing segmentation results with masks
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
        
        # Add segmentation-specific parameters
        if save_masks:
            predict_args['save_masks'] = save_masks

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
        save_masks: bool = False,
        output_dir: Union[str, Path] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    ):
        """
        Perform batch segmentation inference on all images in a directory.

        Args:
            source_dir (str | Path): Directory containing images
            save (bool): Save annotated images with masks
            show (bool): Display segmentation results
            save_txt (bool): Save segmentation results to txt files
            save_conf (bool): Save confidence scores in txt files
            save_crop (bool): Save cropped instance images
            save_masks (bool): Save instance masks as separate images
            output_dir (str | Path): Output directory for saved results
            extensions (tuple): Supported image file extensions

        Returns:
            List of Results objects containing segmentation results for all images
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
        
        # Add segmentation-specific parameters
        if save_masks:
            predict_args['save_masks'] = save_masks

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
        save_masks: bool = False,
        output_dir: Union[str, Path] = None
    ):
        """
        Perform segmentation inference on a list of image paths.

        Args:
            image_list (List[str | Path]): List of image file paths
            save (bool): Save annotated images with masks
            show (bool): Display segmentation results
            save_txt (bool): Save segmentation results to txt files
            save_conf (bool): Save confidence scores in txt files
            save_crop (bool): Save cropped instance images
            save_masks (bool): Save instance masks as separate images
            output_dir (str | Path): Output directory for saved results

        Returns:
            List of Results objects containing segmentation results for all images
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
                save_masks=save_masks,
                output_dir=output_dir
            )
            all_results.extend(results)

        return all_results

    def print_results_summary(self, results, source_info: str = ""):
        """Print a summary of segmentation results."""
        if not results:
            LOGGER.info(f"No results to display{f' for {source_info}' if source_info else ''}")
            return

        total_instances = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
        total_masks = sum(len(r.masks.data) if r.masks is not None else 0 for r in results)
        
        if self.verbose:
            LOGGER.info(f"\n{colorstr('Segmentation Results Summary')}{f' for {source_info}' if source_info else ''}:")
            LOGGER.info(f"  Images processed: {len(results)}")
            LOGGER.info(f"  Total instances: {total_instances}")
            LOGGER.info(f"  Total masks generated: {total_masks}")

            if hasattr(self.model.model, 'names') and total_instances > 0:
                # Count instances per class
                class_counts = {}
                mask_counts = {}
                
                for result in results:
                    if result.boxes is not None:
                        for cls in result.boxes.cls:
                            cls_name = self.model.model.names[int(cls)]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    if result.masks is not None:
                        for i, cls in enumerate(result.boxes.cls if result.boxes is not None else []):
                            cls_name = self.model.model.names[int(cls)]
                            mask_counts[cls_name] = mask_counts.get(cls_name, 0) + 1
                
                LOGGER.info("  Instances by class:")
                for cls_name, count in sorted(class_counts.items()):
                    mask_count = mask_counts.get(cls_name, 0)
                    LOGGER.info(f"    {cls_name}: {count} instances ({mask_count} masks)")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv12-DINO Segmentation Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image segmentation
    python inference.py --weights best.pt --source image.jpg --save --show

    # Batch segmentation on directory
    python inference.py --weights runs/segment/train/weights/best.pt --source test_images/ --output results/

    # Custom thresholds and save options with masks
    python inference.py --weights model.pt --source images/ --conf 0.5 --iou 0.7 --save-txt --save-masks

    # Complete segmentation output
    python inference.py --weights model.pt --source image.jpg --save --save-masks --save-crop --show
        """
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help='Path to trained segmentation model weights (.pt file)'
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
        help='Output directory for results (default: runs/segment/predict)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for instance detection (default: 0.25)'
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
        help='Save annotated images with segmentation masks'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Display segmentation results'
    )

    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save segmentation results to txt files'
    )

    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidence scores in txt files'
    )

    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='Save cropped instance images'
    )

    parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save instance masks as separate images'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )

    return parser.parse_args()


def main():
    """Main segmentation inference function."""
    args = parse_arguments()

    try:
        # Initialize segmentation inference model
        inference = YOLOSegmentationInference(
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
            # Single image segmentation
            if args.verbose:
                LOGGER.info(f"Running segmentation inference on single image: {source_path}")
            
            results = inference.predict_single(
                source=source_path,
                save=args.save,
                show=args.show,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_crop=args.save_crop,
                save_masks=args.save_masks,
                output_dir=args.output
            )
            
            inference.print_results_summary(results, str(source_path))

        elif source_path.is_dir():
            # Directory batch segmentation
            if args.verbose:
                LOGGER.info(f"Running batch segmentation inference on directory: {source_path}")
            
            results = inference.predict_batch(
                source_dir=source_path,
                save=args.save,
                show=args.show,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_crop=args.save_crop,
                save_masks=args.save_masks,
                output_dir=args.output
            )
            
            inference.print_results_summary(results, str(source_path))

        else:
            raise FileNotFoundError(f"Source not found: {source_path}")

        # Print timing information
        end_time = time.time()
        if args.verbose:
            LOGGER.info(f"\nSegmentation inference completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        LOGGER.error(f"Error during segmentation inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()