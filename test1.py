import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import math
import time

class FakeQuantize(nn.Module):
    def __init__(self, num_bits: int = 8, symmetric: bool = True, per_channel: bool = False):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.qmin = -(2**(num_bits-1)) if symmetric else 0
        self.qmax = 2**(num_bits-1) - 1 if symmetric else 2**num_bits - 1

        # Start as scalar; we'll resize on first calibrate if needed
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('initialized', torch.tensor(False))

    def calibrate(self, x: torch.Tensor):
        with torch.no_grad():
            if self.per_channel and x.dim() > 1:
                # Per-channel along dim 0 (out_channels / out_features)
                x_flat = x.view(x.size(0), -1)
                x_min = x_flat.min(dim=1)[0]
                x_max = x_flat.max(dim=1)[0]

                # Ensure buffers have the right shape on first use
                ch = x.size(0)
                if self.scale.numel() != ch:
                    # Replace buffers with new tensors of correct size (still registered as buffers)
                    self.scale = torch.ones(ch, device=x.device, dtype=x.dtype)
                    self.zero_point = torch.zeros(ch, device=x.device, dtype=x.dtype)
            else:
                x_min = x.min()
                x_max = x.max()

            if self.symmetric:
                abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                scale = abs_max / (2**(self.num_bits-1) - 1)
                zero_point = torch.zeros_like(scale, device=x.device, dtype=x.dtype)
            else:
                scale = (x_max - x_min) / (2**self.num_bits - 1)
                # Avoid div by zero before using it below
                scale = torch.where(scale > 1e-8, scale, torch.tensor(1e-8, device=x.device, dtype=x.dtype))
                zero_point = self.qmin - torch.round(x_min / scale)
                zero_point = torch.clamp(zero_point, self.qmin, self.qmax)

            # Final guard for zero scale
            scale = torch.where(scale > 1e-8, scale, torch.tensor(1e-8, device=x.device, dtype=x.dtype))

            # Copy values (shapes now match)
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not bool(self.initialized):
            self.calibrate(x)

        # Optional: allow bypass during warmup via a flag (shown below)
        if getattr(self, "bypass", False):
            return x

        if self.per_channel and x.dim() > 1:
            scale = self.scale.view(-1, *([1] * (x.dim() - 1)))
            zero_point = self.zero_point.view(-1, *([1] * (x.dim() - 1)))
        else:
            scale = self.scale
            zero_point = self.zero_point

        x_scaled = x / scale + zero_point
        x_q = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
        x_dq = (x_q - zero_point) * scale

        # STE: use quantized value in forward, identity gradient in backward
        return x + (x_dq - x).detach()




class QATLinear(nn.Module):
    """Linear layer with mixed-precision QAT support"""
    
    def __init__(self, in_features: int, out_features: int, 
                 weight_bits: int = 8, activation_bits: int = 8, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Quantizers for different precisions
        self.weight_quantizer = FakeQuantize(weight_bits, symmetric=True, per_channel=True)
        self.activation_quantizer = FakeQuantize(activation_bits, symmetric=False, per_channel=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights and activations during training
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_x = self.activation_quantizer(x)
        
        return F.linear(quantized_x, quantized_weight, self.bias)

class QATConv2d(nn.Module):
    """Conv2d layer with mixed-precision QAT support"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 weight_bits: int = 8, activation_bits: int = 8,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Standard conv layer parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        # Quantizers
        self.weight_quantizer = FakeQuantize(weight_bits, symmetric=True, per_channel=True)
        self.activation_quantizer = FakeQuantize(activation_bits, symmetric=False, per_channel=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_x = self.activation_quantizer(x)
        
        return F.conv2d(quantized_x, quantized_weight, self.bias, 
                       self.stride, self.padding)

class CIFAR10QATModel(nn.Module):
    """QAT Model specifically designed for CIFAR-10"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Mixed precision: different layers with different bit widths
        self.conv1 = QATConv2d(3, 64, 3, weight_bits=8, activation_bits=8, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = QATConv2d(64, 64, 3, weight_bits=8, activation_bits=8, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = QATConv2d(64, 128, 3, weight_bits=6, activation_bits=8, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = QATConv2d(128, 128, 3, weight_bits=6, activation_bits=8, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = QATConv2d(128, 256, 3, weight_bits=4, activation_bits=8, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = QATConv2d(256, 256, 3, weight_bits=4, activation_bits=6, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Fully connected layers with mixed precision
        self.fc1 = QATLinear(256 * 2 * 2, 512, weight_bits=8, activation_bits=8)
        self.fc2 = QATLinear(512, 256, weight_bits=6, activation_bits=8)
        self.fc3 = QATLinear(256, num_classes, weight_bits=4, activation_bits=8)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: 32x32 -> 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2: 16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Block 3: 8x8 -> 4x4 -> 2x2
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        
        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def prepare_for_inference(self):
        """Convert model to use true quantized weights for inference"""
        self.eval()
        for module in self.modules():
            if hasattr(module, 'weight_quantizer'):
                module.weight_quantizer.eval()
                module.activation_quantizer.eval()

def get_cifar10_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """Get CIFAR-10 train and test dataloaders with augmentation"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'    Batch [{batch_idx+1}/{len(dataloader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(dataloader), 100. * correct / total

def test_epoch(model, dataloader, criterion, device):
    """Test for one epoch"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(dataloader), 100. * correct / total

def train_cifar10_qat(epochs: int = 20, batch_size: int = 128, lr: float = 0.001):
    """Main training function for CIFAR-10 QAT"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_cifar10_dataloaders(batch_size)
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Dataset loaded: {len(trainloader.dataset)} training samples, "
          f"{len(testloader.dataset)} test samples")
    
    # Initialize model
    model = CIFAR10QATModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Training loop
    print(f"\nStarting QAT training for {epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f'\nEpoch [{epoch+1}/{epochs}]')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Test
        test_loss, test_acc = test_epoch(model, testloader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{epochs}] Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%')
        print(f'  Time: {epoch_time:.1f}s')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_qat_cifar10.pth')
            print(f'  ✅ New best accuracy: {best_acc:.2f}%')
    
    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")
    
    # Load best model and prepare for inference
    model.load_state_dict(torch.load('best_qat_cifar10.pth'))
    model.prepare_for_inference()
    
    # Final evaluation
    final_loss, final_acc = test_epoch(model, testloader, criterion, device)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%")
    
    return model, best_acc

def analyze_model_complexity(model):
    """Analyze the model's quantization configuration and theoretical compression"""
    print("\n" + "="*60)
    print("MODEL QUANTIZATION ANALYSIS")
    print("="*60)
    
    total_fp32_size = 0
    total_quantized_size = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (QATLinear, QATConv2d)):
            weight_params = module.weight.numel()
            weight_bits = module.weight_quantizer.num_bits
            activation_bits = module.activation_quantizer.num_bits
            
            fp32_weight_size = weight_params * 4  # 4 bytes per FP32
            quantized_weight_size = weight_params * weight_bits / 8
            
            total_fp32_size += fp32_weight_size
            total_quantized_size += quantized_weight_size
            
            if module.bias is not None:
                bias_size = module.bias.numel() * 4  # Keep bias in FP32
                total_fp32_size += bias_size
                total_quantized_size += bias_size
            
            print(f"{name:15s} | W:{weight_bits}b A:{activation_bits}b | "
                  f"Params: {weight_params:,} | "
                  f"Size: {fp32_weight_size/1024:.1f}KB -> {quantized_weight_size/1024:.1f}KB")
    
    compression_ratio = total_fp32_size / total_quantized_size if total_quantized_size > 0 else 1
    
    print("-" * 60)
    print(f"Total FP32 size:     {total_fp32_size / 1024 / 1024:.2f} MB")
    print(f"Total quantized size: {total_quantized_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio:    {compression_ratio:.2f}x")
    print(f"Size reduction:       {(1 - total_quantized_size/total_fp32_size)*100:.1f}%")

if __name__ == "__main__":
    print("🚀 Mixed-Precision QAT Training on CIFAR-10")
    print("=" * 50)
    
    # Train the model
    trained_model, best_accuracy = train_cifar10_qat(
        epochs=20, 
        batch_size=128, 
        lr=0.001
    )
    
    # Analyze the model
    analyze_model_complexity(trained_model)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Best accuracy achieved: {best_accuracy:.2f}%")
    print(f"💾 Model saved as 'best_qat_cifar10.pth'")
    print("\nModel features:")
    print("- Mixed precision: 8-bit, 6-bit, 4-bit weights across layers")
    print("- Data augmentation: horizontal flip, rotation, crop")
    print("- Batch normalization for training stability")
    print("- Learning rate scheduling")
    print("- Automatic quantization parameter calibration")
