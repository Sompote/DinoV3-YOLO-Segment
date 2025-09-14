#!/usr/bin/env python3
"""
Training Variant Testing Suite

This script tests different DINOv3 variants and options through actual training runs
to validate real-world performance and stability.

Usage:
    python training_variant_tester.py --quick-test
    python training_variant_tester.py --full-matrix
    python training_variant_tester.py --benchmark-configs
    python training_variant_tester.py --custom-dataset ./my_data.yaml
"""

import argparse
import subprocess
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import yaml

@dataclass
class TrainingConfig:
    """Training configuration"""
    name: str
    yolo_size: str
    dino_input: Optional[str]
    integration: str
    epochs: int
    batch_size: int
    expected_improvement: Optional[float] = None
    description: str = ""

@dataclass
class TrainingResult:
    """Training result"""
    config: TrainingConfig
    success: bool
    execution_time: float
    final_map: Optional[float] = None
    final_loss: Optional[float] = None
    max_memory: Optional[float] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None

class TrainingVariantTester:
    """Training-based variant testing"""
    
    def __init__(self, data_path: str = "coco.yaml", device: str = "0"):
        self.data_path = data_path
        self.device = device
        self.results: List[TrainingResult] = []
        self.base_dir = Path("training_tests")
        self.base_dir.mkdir(exist_ok=True)
        
        # Predefined test configurations
        self.quick_configs = [
            TrainingConfig(
                name="baseline_yolov12s",
                yolo_size="s",
                dino_input=None,
                integration="single",
                epochs=10,
                batch_size=16,
                description="Baseline YOLOv12-Small without DINO"
            ),
            TrainingConfig(
                name="dinov3_vitb16_recommended",
                yolo_size="s", 
                dino_input="dinov3_vitb16",
                integration="single",
                epochs=10,
                batch_size=8,
                expected_improvement=5.0,
                description="Recommended DINOv3 ViT-B/16 configuration"
            ),
            TrainingConfig(
                name="dinov3_convnext_hybrid",
                yolo_size="m",
                dino_input="dinov3_convnext_base",
                integration="single", 
                epochs=10,
                batch_size=6,
                expected_improvement=7.0,
                description="ConvNeXt hybrid CNN-ViT architecture"
            ),
            TrainingConfig(
                name="dinov3_dual_scale",
                yolo_size="l",
                dino_input="dinov3_vitl16",
                integration="dual",
                epochs=5,  # Reduced for dual-scale
                batch_size=4,
                expected_improvement=10.0,
                description="High-performance dual-scale integration"
            )
        ]
        
        self.benchmark_configs = [
            # Small models comparison
            TrainingConfig("yolo_s_baseline", "s", None, "single", 20, 16, description="YOLOv12s baseline"),
            TrainingConfig("yolo_s_dino_small", "s", "dinov3_vits16", "single", 20, 12, description="YOLOv12s + DINOv3-Small"),
            TrainingConfig("yolo_s_dino_base", "s", "dinov3_vitb16", "single", 20, 8, description="YOLOv12s + DINOv3-Base"),
            
            # Medium models comparison  
            TrainingConfig("yolo_m_baseline", "m", None, "single", 15, 12, description="YOLOv12m baseline"),
            TrainingConfig("yolo_m_convnext", "m", "dinov3_convnext_base", "single", 15, 6, description="YOLOv12m + ConvNeXt"),
            
            # Large models comparison
            TrainingConfig("yolo_l_baseline", "l", None, "single", 10, 8, description="YOLOv12l baseline"),
            TrainingConfig("yolo_l_dino_large", "l", "dinov3_vitl16", "single", 10, 4, description="YOLOv12l + DINOv3-Large"),
            TrainingConfig("yolo_l_dual_scale", "l", "dinov3_vitl16", "dual", 10, 2, description="YOLOv12l + Dual-scale"),
        ]
        
        self.stress_test_configs = [
            # Memory stress tests
            TrainingConfig("stress_huge_model", "x", "dinov3_vith16plus", "single", 5, 1, description="Huge model stress test"),
            TrainingConfig("stress_dual_large", "l", "dinov3_vitl16", "dual", 5, 1, description="Dual-scale stress test"),
            
            # Batch size variations
            TrainingConfig("batch_test_small", "s", "dinov3_vitb16", "single", 5, 32, description="High batch size test"),
            TrainingConfig("batch_test_large", "l", "dinov3_vitl16", "single", 5, 8, description="Large model batch test"),
        ]
    
    def create_training_command(self, config: TrainingConfig) -> List[str]:
        """Create training command for configuration"""
        cmd = [
            "python", "train_yolov12_dino.py",
            "--data", self.data_path,
            "--yolo-size", config.yolo_size,
            "--epochs", str(config.epochs),
            "--batch-size", str(config.batch_size),
            "--device", self.device,
            "--name", config.name,
            "--save-period", "5"  # Save checkpoints every 5 epochs
        ]
        
        if config.dino_input:
            cmd.extend(["--dino-version", "3"])
            cmd.extend(["--dino-input", config.dino_input])
            cmd.extend(["--integration", config.integration])
        
        return cmd
    
    def run_training(self, config: TrainingConfig) -> TrainingResult:
        """Run training for a single configuration"""
        print(f"\n🏋️  Training: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Command: {config.yolo_size} + {config.dino_input or 'baseline'} ({config.integration})")
        
        result = TrainingResult(config, False, 0.0)
        start_time = time.time()
        
        try:
            # Create training command
            cmd = self.create_training_command(config)
            print(f"   Executing: {' '.join(cmd)}")
            
            # Create log file
            log_file = self.base_dir / f"{config.name}_training.log"
            result.log_file = str(log_file)
            
            # Run training
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output and save to log
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    
                    # Print progress indicators
                    if "Epoch" in line:
                        print(f"   {line.strip()}")
                    elif "mAP" in line:
                        print(f"   {line.strip()}")
                
                process.wait()
            
            # Check training success
            if process.returncode == 0:
                result.success = True
                print(f"   ✅ Training completed successfully")
                
                # Parse results from log
                self.parse_training_results(result)
                
            else:
                result.error_message = f"Training failed with return code {process.returncode}"
                print(f"   ❌ Training failed: {result.error_message}")
        
        except subprocess.TimeoutExpired:
            result.error_message = "Training timeout"
            print(f"   ⚠️  Training timeout")
        except Exception as e:
            result.error_message = str(e)
            print(f"   ❌ Training error: {e}")
        
        finally:
            result.execution_time = time.time() - start_time
            print(f"   ⏱️  Execution time: {result.execution_time:.1f}s")
        
        return result
    
    def parse_training_results(self, result: TrainingResult):
        """Parse training results from log file"""
        if not result.log_file or not os.path.exists(result.log_file):
            return
        
        try:
            with open(result.log_file, 'r') as f:
                content = f.read()
            
            # Extract final mAP (look for last occurrence)
            lines = content.split('\n')
            for line in reversed(lines):
                if 'mAP@0.5' in line or 'mAP50' in line:
                    # Try to extract mAP value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'mAP' in part and i + 1 < len(parts):
                            try:
                                result.final_map = float(parts[i + 1])
                                break
                            except ValueError:
                                continue
                    if result.final_map:
                        break
            
            # Extract final loss
            for line in reversed(lines):
                if 'loss' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'loss' in part.lower() and i + 1 < len(parts):
                            try:
                                result.final_loss = float(parts[i + 1])
                                break
                            except ValueError:
                                continue
                    if result.final_loss:
                        break
        
        except Exception as e:
            print(f"   ⚠️  Failed to parse training results: {e}")
    
    def run_quick_test(self) -> List[TrainingResult]:
        """Run quick validation test"""
        print("🚀 Running Quick Training Test")
        print("=" * 50)
        
        results = []
        for config in self.quick_configs:
            result = self.run_training(config)
            results.append(result)
            self.results.append(result)
            
            # Brief pause between trainings
            time.sleep(2)
        
        return results
    
    def run_benchmark_matrix(self) -> List[TrainingResult]:
        """Run comprehensive benchmark matrix"""
        print("📊 Running Benchmark Matrix")
        print("=" * 50)
        
        results = []
        for config in self.benchmark_configs:
            result = self.run_training(config)
            results.append(result)
            self.results.append(result)
            
            # Pause between trainings
            time.sleep(5)
        
        return results
    
    def run_stress_tests(self) -> List[TrainingResult]:
        """Run stress tests for edge cases"""
        print("🔥 Running Stress Tests")
        print("=" * 50)
        
        results = []
        for config in self.stress_test_configs:
            result = self.run_training(config)
            results.append(result)
            self.results.append(result)
            
            # Longer pause for stress tests
            time.sleep(10)
        
        return results
    
    def compare_results(self, baseline_name: str, enhanced_name: str) -> Dict[str, Any]:
        """Compare baseline vs enhanced results"""
        baseline = next((r for r in self.results if r.config.name == baseline_name), None)
        enhanced = next((r for r in self.results if r.config.name == enhanced_name), None)
        
        if not baseline or not enhanced:
            return {"error": "Missing baseline or enhanced results"}
        
        comparison = {
            "baseline": {
                "name": baseline.config.name,
                "map": baseline.final_map,
                "loss": baseline.final_loss,
                "time": baseline.execution_time
            },
            "enhanced": {
                "name": enhanced.config.name, 
                "map": enhanced.final_map,
                "loss": enhanced.final_loss,
                "time": enhanced.execution_time
            }
        }
        
        # Calculate improvements
        if baseline.final_map and enhanced.final_map:
            comparison["map_improvement"] = enhanced.final_map - baseline.final_map
            comparison["map_improvement_pct"] = ((enhanced.final_map - baseline.final_map) / baseline.final_map) * 100
        
        if baseline.final_loss and enhanced.final_loss:
            comparison["loss_improvement"] = baseline.final_loss - enhanced.final_loss
        
        comparison["time_overhead"] = enhanced.execution_time - baseline.execution_time
        comparison["time_overhead_pct"] = ((enhanced.execution_time - baseline.execution_time) / baseline.execution_time) * 100
        
        return comparison
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        report = {
            "summary": {
                "total_trainings": len(self.results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": (len(successful_results) / len(self.results) * 100) if self.results else 0
            },
            "performance_analysis": {},
            "comparisons": {},
            "recommendations": [],
            "failed_trainings": [asdict(r) for r in failed_results],
            "all_results": [asdict(r) for r in self.results]
        }
        
        # Performance analysis
        if successful_results:
            maps = [r.final_map for r in successful_results if r.final_map is not None]
            times = [r.execution_time for r in successful_results]
            
            report["performance_analysis"] = {
                "avg_map": sum(maps) / len(maps) if maps else None,
                "best_map": max(maps) if maps else None,
                "worst_map": min(maps) if maps else None,
                "avg_training_time": sum(times) / len(times),
                "fastest_training": min(times),
                "slowest_training": max(times)
            }
        
        # Generate comparisons
        comparisons = [
            ("yolo_s_baseline", "yolo_s_dino_base"),
            ("yolo_m_baseline", "yolo_m_convnext"),
            ("yolo_l_baseline", "yolo_l_dino_large"),
        ]
        
        for baseline, enhanced in comparisons:
            comparison = self.compare_results(baseline, enhanced)
            if "error" not in comparison:
                report["comparisons"][f"{baseline}_vs_{enhanced}"] = comparison
        
        # Generate recommendations
        if successful_results:
            # Best performing configuration
            best_map_result = max(successful_results, key=lambda r: r.final_map or 0)
            report["recommendations"].append(f"Best mAP: {best_map_result.config.name} ({best_map_result.final_map:.2f}%)")
            
            # Fastest training
            fastest_result = min(successful_results, key=lambda r: r.execution_time)
            report["recommendations"].append(f"Fastest training: {fastest_result.config.name} ({fastest_result.execution_time:.1f}s)")
            
            # Best efficiency (mAP/time ratio)
            efficiency_results = [(r.final_map / r.execution_time, r) for r in successful_results if r.final_map]
            if efficiency_results:
                best_efficiency = max(efficiency_results, key=lambda x: x[0])[1]
                report["recommendations"].append(f"Best efficiency: {best_efficiency.config.name}")
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted training report"""
        print("\n" + "=" * 80)
        print("📊 TRAINING VARIANT TESTING REPORT")
        print("=" * 80)
        
        # Summary
        summary = report["summary"]
        print(f"\n📋 Training Summary:")
        print(f"   Total Trainings: {summary['total_trainings']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        # Performance analysis
        if report["performance_analysis"]:
            perf = report["performance_analysis"]
            print(f"\n⚡ Performance Analysis:")
            if perf["avg_map"]:
                print(f"   Average mAP: {perf['avg_map']:.2f}%")
                print(f"   Best mAP: {perf['best_map']:.2f}%")
                print(f"   Worst mAP: {perf['worst_map']:.2f}%")
            print(f"   Average Training Time: {perf['avg_training_time']:.1f}s")
            print(f"   Fastest/Slowest: {perf['fastest_training']:.1f}s / {perf['slowest_training']:.1f}s")
        
        # Comparisons
        if report["comparisons"]:
            print(f"\n🔄 Baseline vs Enhanced Comparisons:")
            for name, comparison in report["comparisons"].items():
                if "map_improvement" in comparison:
                    print(f"   {name}:")
                    print(f"     mAP Improvement: +{comparison['map_improvement']:.2f}% ({comparison['map_improvement_pct']:+.1f}%)")
                    print(f"     Time Overhead: +{comparison['time_overhead']:.1f}s ({comparison['time_overhead_pct']:+.1f}%)")
        
        # Recommendations
        if report["recommendations"]:
            print(f"\n🎯 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        # Failed trainings
        if report["failed_trainings"]:
            print(f"\n❌ Failed Trainings:")
            for failure in report["failed_trainings"]:
                print(f"   • {failure['config']['name']}: {failure['error_message']}")
    
    def save_report(self, filename: Optional[str] = None):
        """Save training report to file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"training_test_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {filename}")
        return filename

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training Variant Testing Suite')
    
    # Test modes
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick validation test (4 configs, 10 epochs each)')
    parser.add_argument('--benchmark-matrix', action='store_true', 
                       help='Run comprehensive benchmark matrix')
    parser.add_argument('--stress-tests', action='store_true',
                       help='Run stress tests for edge cases')
    parser.add_argument('--full-matrix', action='store_true',
                       help='Run all test suites')
    
    # Configuration
    parser.add_argument('--data', type=str, default='coco.yaml',
                       help='Dataset YAML file')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device')
    parser.add_argument('--save-report', type=str, default=None,
                       help='Save report to file')
    
    return parser.parse_args()

def main():
    """Main training testing function"""
    args = parse_arguments()
    
    print("🏋️  Training Variant Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = TrainingVariantTester(args.data, args.device)
    
    try:
        # Run selected tests
        if args.full_matrix:
            tester.run_quick_test()
            tester.run_benchmark_matrix()
            tester.run_stress_tests()
        else:
            if args.quick_test:
                tester.run_quick_test()
            
            if args.benchmark_matrix:
                tester.run_benchmark_matrix()
            
            if args.stress_tests:
                tester.run_stress_tests()
            
            # Default to quick test if no options specified
            if not any([args.quick_test, args.benchmark_matrix, args.stress_tests]):
                tester.run_quick_test()
        
        # Generate and print report
        report = tester.generate_report()
        tester.print_report(report)
        
        # Save report
        if args.save_report:
            tester.save_report(args.save_report)
        
        # Exit with appropriate code
        if report["summary"]["success_rate"] >= 75:
            print("\n🎉 Training testing completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Training testing completed with issues")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(2)

if __name__ == '__main__':
    main()