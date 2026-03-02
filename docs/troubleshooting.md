# Troubleshooting Jarvis AI

## PyTorch DLL Issues on Windows

If you encounter a PyTorch DLL error on Windows (e.g., missing DLLs or import failures), reinstall the CPU-only version of PyTorch using:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This resolves most DLL-related issues for Windows users. After installation, rerun your training or inference script.

## Other Common Issues

- See the main README and DEPLOYMENT_GUIDE for additional troubleshooting tips.
- For CUDA/GPU issues, ensure your drivers are up to date or use the CPU-only install as above.
- For missing dependencies, run:
  ```bash
  pip install -r requirements.txt
  ```
- For more help, open an issue on GitHub or check the Discussions board.
