---
# Jarvis AI Platform - Quickstart & Onboarding

Welcome to Jarvis! This guide will help you get started, understand the platform, and onboard new contributors.

## 🚀 Quickstart

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Jarvis
   ```
2. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in secrets as needed.
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the platform:**
   ```sh
   python api.py
   ```
   Or with Docker:
   ```sh
   docker build -t jarvis .
   docker run -p 8000:8000 --env-file .env jarvis
   ```
5. **Access the API/docs:**
   - Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.
   - Admin dashboard: `/admin` (admin login required)
   - No-code workflow editor: `/workflow`

## 🧑‍💻 Onboarding for Contributors

- **Read the `README.md` and `QUICKSTART.md` for project structure and goals.**
- **Check `HOW_TO_TEST.md` for testing instructions.**
- **Review `SECURITY.md`, `PRIVACY_POLICY.md`, and `ACCESSIBILITY_STATEMENT.md` for compliance.**
- **See `docs/` for detailed guides and architecture.**
- **Use the issue and PR templates for contributions.**
- **Join the community via the plugin registry and `/plugins` endpoints.**

## 🛠️ Key Features
- Model versioning, rollback, and registry
- GPU/accelerator-aware serving
- External model server integration
- Plugin/extension marketplace
- Drift detection and monitoring
- Audit trail and compliance dashboard
- Real-time collaboration and annotation
- Self-healing and auto-scaling
- Advanced security (API key, RBAC, SSO)
- No-code/low-code workflow editor

## 🧩 Extending Jarvis
- Add new models, plugins, or endpoints via the `/plugins` API or by contributing to the codebase.
- See `REGISTRY.md` and `HACKATHON_TEMPLATE.md` for community and hackathon contributions.

## 💬 Need Help?
- Open an issue or discussion on GitHub.
- Contact the maintainers via the email in `README.md`.

Welcome to the future of AI platforms! 🚀
- **Train R²**: How well the model fits training data (higher is better, max 1.0)
- **Val R²**: How well the model generalizes to new data (higher is better)
- **Train/Val MSE**: Mean squared error (lower is better)

Example good results:
```
Train R²: 0.9812  ← Excellent fit to training data
Val R²: 0.9686    ← Great generalization
Train MSE: 0.1129 ← Low error
Val MSE: 0.1937   ← Reasonable error on new data
```

## 🔧 Advanced Usage

### Direct Python API
```python
from src.models.numpy_neural_network import SimpleNeuralNetwork
from src.data.numpy_processor import DataProcessor

# Create and train model
model = SimpleNeuralNetwork(input_size=10, hidden_sizes=[64, 32], output_size=1)
processor = DataProcessor()

# Load and process data
X_train, X_test, y_train, y_test = processor.process_pipeline("your_data.csv")

# Train
model.fit(X_train, y_train, epochs=100)

# Predict
predictions = model.predict(X_test)
```

### Command Line Training
```bash
python -m src.training.train_final --config config/train_config.yaml
```

### Command Line Inference
```bash
python -m src.inference.predict --model models/trained_model.pkl
```

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
python jarvis.py install
```

### "No such file or directory"
Make sure you're in the Jarvis project folder when running commands.

### "Permission denied" (Windows)
This is normal for some tests. The main functionality still works.

### Poor model performance
- Try more epochs: Change `epochs: 200` to `epochs: 500` in config
- Try different architecture: Change `hidden_sizes: [64, 32]` to `[128, 64, 32]`
- Check your data: Make sure target column is numerical

## 📞 Need Help?

1. Check that you're in the right directory (should contain `jarvis.py`)
2. Make sure Python 3.8+ is installed: `python --version`
3. Try the install command again: `python jarvis.py install`
4. Run the tests: `python jarvis.py test`

## 🎉 What's Next?

- Try different neural network architectures
- Experiment with your own datasets
- Modify the code to add new features
- Deploy your model to production!

---

**Ready to build some AI? Let's go!** 🤖✨
