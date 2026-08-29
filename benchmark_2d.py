import torch
import time
from models_2d import ResNet18_2D, STFTRADN

def measure_latency(model, input_shape, num_runs=100, warmup_runs=10):
    """Measures the inference latency per sample in milliseconds."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            
    # Measure
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    latency_per_sample_ms = total_time_ms / (num_runs * input_shape[0])
    return latency_per_sample_ms

def count_trainable_parameters(model):
    """Calculates the total trainable parameter footprint."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def export_to_onnx(model, input_shape, filename="stft_radn_model.onnx"):
    """Exports the model weights to an ONNX file."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[INFO] Model exported successfully to {filename}")

if __name__ == "__main__":
    # Standard input shape: [Batch, Channels, Height, Width]
    # Corresponding to [1, 1, 64, 5] for grayscale 2D spectrograms
    input_shape = (1, 1, 64, 5)
    
    print("="*60)
    print("Initializing Models...")
    resnet_model = ResNet18_2D(num_classes=11, in_channels=1)
    stft_model = STFTRADN(num_classes=11, in_channels=1)
    
    print("="*60)
    print(" ResNet18_2D Benchmarks")
    print("="*60)
    resnet_params = count_trainable_parameters(resnet_model)
    print(f"Trainable Parameters : {resnet_params:,}")
    resnet_latency = measure_latency(resnet_model, input_shape)
    print(f"Inference Latency    : {resnet_latency:.4f} ms / sample")
    
    print("\n" + "="*60)
    print(" STFTRADN Benchmarks")
    print("="*60)
    stft_params = count_trainable_parameters(stft_model)
    print(f"Trainable Parameters : {stft_params:,}")
    stft_latency = measure_latency(stft_model, input_shape)
    print(f"Inference Latency    : {stft_latency:.4f} ms / sample")
    
    print("\n" + "="*60)
    print(" ONNX Export (STFTRADN for Grad-CAM validation)")
    print("="*60)
    export_to_onnx(stft_model, input_shape, filename="stft_radn_model.onnx")
    print("="*60)
