"""
PyTorch → ONNX → CoreML Production Pipeline
=============================================
End-to-end ML pipeline for training EfficientNet-B0 with transfer learning,
exporting to ONNX format, and optimizing with INT8 quantization.

Author: SQ1111
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
class PipelineConfig:
    """Central configuration for the entire ML pipeline."""
    
    # Model
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 10
    PRETRAINED = True
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    USE_AMP = True  # Automatic Mixed Precision
    
    # Data
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    
    # Export
    ONNX_OPSET = 17
    OUTPUT_DIR = Path("outputs")
    
    # Device
    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Data Pipeline
# ============================================================
class DataPipeline:
    """Handles data loading and augmentation."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE + 32),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def get_dataloaders(self, data_dir: str):
        """Create train and validation dataloaders."""
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"),
            transform=self.train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "val"),
            transform=self.val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
        return train_loader, val_loader


# ============================================================
# Model Builder
# ============================================================
class ModelBuilder:
    """Builds and configures the EfficientNet model."""
    
    @staticmethod
    def build(config: PipelineConfig) -> nn.Module:
        """Build EfficientNet-B0 with custom classification head."""
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if config.PRETRAINED else None
        )
        
        # Freeze backbone for transfer learning
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Custom classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, config.NUM_CLASSES),
        )
        
        return model.to(config.DEVICE)
    
    @staticmethod
    def unfreeze_backbone(model: nn.Module, layers: int = 3):
        """Gradually unfreeze backbone layers for fine-tuning."""
        children = list(model.features.children())
        for child in children[-layers:]:
            for param in child.parameters():
                param.requires_grad = True
        return model


# ============================================================
# Training Engine
# ============================================================
class TrainingEngine:
    """Handles model training with AMP and learning rate scheduling."""
    
    def __init__(self, model, config: PipelineConfig):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )
        self.scaler = GradScaler(enabled=config.USE_AMP)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    def train_epoch(self, dataloader):
        """Train for one epoch with AMP."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.config.DEVICE)
            targets = targets.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        for images, targets in dataloader:
            images = images.to(self.config.DEVICE)
            targets = targets.to(self.config.DEVICE)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """Full training loop with logging."""
        best_acc = 0.0
        
        print(f"\n{'='*60}")
        print(f"  Training on {self.config.DEVICE.upper()}")
        print(f"  Model: {self.config.MODEL_NAME}")
        print(f"  Epochs: {self.config.EPOCHS} | Batch: {self.config.BATCH_SIZE}")
        print(f"  AMP: {self.config.USE_AMP}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.EPOCHS + 1):
            start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            marker = " ★ BEST" if val_acc > best_acc else ""
            print(
                f"  Epoch {epoch:3d}/{self.config.EPOCHS} │ "
                f"Loss: {train_loss:.4f} / {val_loss:.4f} │ "
                f"Acc: {val_acc:.1f}% │ "
                f"LR: {lr:.2e} │ "
                f"{elapsed:.1f}s{marker}"
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
        
        print(f"\n  Best Accuracy: {best_acc:.1f}%")
        return self.history
    
    def _save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint."""
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "config": {
                "model": self.config.MODEL_NAME,
                "num_classes": self.config.NUM_CLASSES,
                "image_size": self.config.IMAGE_SIZE,
            }
        }
        path = self.config.OUTPUT_DIR / "best_model.pth"
        torch.save(checkpoint, path)


# ============================================================
# ONNX Exporter
# ============================================================
class ONNXExporter:
    """Export PyTorch model to ONNX format with optimization."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def export(self, model: nn.Module, output_path: str = None):
        """Export model to ONNX with dynamic batch size."""
        model.eval()
        output_path = output_path or str(
            self.config.OUTPUT_DIR / "model.onnx"
        )
        
        dummy_input = torch.randn(
            1, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE,
            device=self.config.DEVICE
        )
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=self.config.ONNX_OPSET,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  ✓ ONNX exported: {output_path} ({file_size:.1f} MB)")
        return output_path
    
    @staticmethod
    def validate(onnx_path: str):
        """Validate ONNX model structure."""
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✓ ONNX validation passed")
        return True


# ============================================================
# INT8 Quantization Engine
# ============================================================
class QuantizationEngine:
    """Quantize ONNX model to INT8 for production deployment."""
    
    @staticmethod
    def quantize_dynamic(onnx_path: str, output_path: str = None):
        """Apply dynamic INT8 quantization."""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        output_path = output_path or onnx_path.replace(".onnx", "_int8.onnx")
        
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"\n  ✓ Quantization complete:")
        print(f"    Original:  {original_size:.1f} MB")
        print(f"    Quantized: {quantized_size:.1f} MB")
        print(f"    Reduction: {reduction:.0f}%")
        
        return output_path


# ============================================================
# Benchmark Suite
# ============================================================
class BenchmarkSuite:
    """Benchmark inference latency and throughput."""
    
    @staticmethod
    def benchmark_pytorch(model, config, num_runs=100):
        """Benchmark PyTorch inference."""
        model.eval()
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                           device=config.DEVICE)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(dummy)
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies.sort()
        return {
            "framework": "PyTorch",
            "device": config.DEVICE,
            "p50_ms": round(latencies[len(latencies)//2], 2),
            "p95_ms": round(latencies[int(len(latencies)*0.95)], 2),
            "p99_ms": round(latencies[int(len(latencies)*0.99)], 2),
            "mean_ms": round(sum(latencies)/len(latencies), 2),
            "throughput_fps": round(1000 / (sum(latencies)/len(latencies)), 1),
        }
    
    @staticmethod
    def benchmark_onnx(onnx_path: str, image_size=224, num_runs=100):
        """Benchmark ONNX Runtime inference."""
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path)
        dummy = {
            "input": torch.randn(1, 3, image_size, image_size).numpy()
        }
        
        # Warmup
        for _ in range(10):
            session.run(None, dummy)
        
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, dummy)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies.sort()
        return {
            "framework": "ONNX Runtime",
            "p50_ms": round(latencies[len(latencies)//2], 2),
            "p95_ms": round(latencies[int(len(latencies)*0.95)], 2),
            "p99_ms": round(latencies[int(len(latencies)*0.99)], 2),
            "mean_ms": round(sum(latencies)/len(latencies), 2),
            "throughput_fps": round(1000 / (sum(latencies)/len(latencies)), 1),
        }
    
    @staticmethod
    def print_results(results: list):
        """Pretty-print benchmark comparison."""
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  {'Framework':<18} {'P50':>8} {'P95':>8} {'P99':>8} {'FPS':>8}")
        print(f"  {'─'*50}")
        for r in results:
            print(
                f"  {r['framework']:<18} "
                f"{r['p50_ms']:>6.1f}ms "
                f"{r['p95_ms']:>6.1f}ms "
                f"{r['p99_ms']:>6.1f}ms "
                f"{r['throughput_fps']:>6.0f}/s"
            )
        print(f"{'='*60}\n")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    """Run the complete ML pipeline."""
    config = PipelineConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  MLforge Pipeline — SQ1111")
    print("  PyTorch → ONNX → INT8 Quantization")
    print("="*60)
    
    # Step 1: Build model
    print("\n[1/5] Building EfficientNet-B0...")
    model = ModelBuilder.build(config)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ Parameters: {params:.1f}M | Device: {config.DEVICE}")
    
    # Step 2: Export to ONNX
    print("\n[2/5] Exporting to ONNX...")
    exporter = ONNXExporter(config)
    onnx_path = exporter.export(model)
    
    # Step 3: Quantize
    print("\n[3/5] INT8 Quantization...")
    quantizer = QuantizationEngine()
    int8_path = quantizer.quantize_dynamic(onnx_path)
    
    # Step 4: Benchmark
    print("\n[4/5] Running benchmarks...")
    bench = BenchmarkSuite()
    results = [
        bench.benchmark_pytorch(model, config),
        bench.benchmark_onnx(onnx_path),
        bench.benchmark_onnx(int8_path),
    ]
    results[1]["framework"] = "ONNX FP32"
    results[2]["framework"] = "ONNX INT8"
    bench.print_results(results)
    
    # Step 5: Save report
    print("[5/5] Saving pipeline report...")
    report = {
        "model": config.MODEL_NAME,
        "parameters_millions": params,
        "device": config.DEVICE,
        "benchmarks": results,
        "files": {
            "pytorch": str(config.OUTPUT_DIR / "best_model.pth"),
            "onnx_fp32": onnx_path,
            "onnx_int8": int8_path,
        }
    }
    report_path = config.OUTPUT_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Report saved: {report_path}")
    print(f"\n  Pipeline complete! 🚀\n")


if __name__ == "__main__":
    main()
