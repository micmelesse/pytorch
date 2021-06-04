set -e
echo "testing"
python -c "import torch; print(torch.__version__)"
# pytest test/test_autograd.py
pytest --verbose test/test_autograd.py::TestAutograd::test_anomaly_detect_nan