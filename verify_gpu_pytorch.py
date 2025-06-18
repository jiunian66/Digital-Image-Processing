# verify_gpu_pytorch.py
import torch
import sys

print("=" * 50)
print("GPU PyTorch 验证")
print("=" * 50)

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print(f"PyTorch CUDA版本: {torch.version.cuda}")
    print(f"是否为GPU版本: 是")
else:
    print("PyTorch CUDA版本: 无")
    print(f"是否为GPU版本: 否")

print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print("\n🎉 GPU信息:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    显存: {props.total_memory / 1024 ** 3:.1f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")

    # 测试GPU计算
    print("\n🧪 GPU测试:")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("  ✅ GPU矩阵运算测试通过")
        print(f"  GPU显存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    except Exception as e:
        print(f"  ❌ GPU测试失败: {e}")
else:
    print("\n❌ GPU不可用")
    print("可能的原因:")
    print("1. 安装的仍是CPU版本")
    print("2. CUDA驱动问题")
    print("3. GPU硬件问题")