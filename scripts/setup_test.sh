# scripts/setup_test.sh
#!/bin/bash
# ---------------------------------------
# setup_test.sh - Verify environment reproducibility
# ---------------------------------------

echo "=== 🧠 Environment Information ==="
python --version
echo

echo "=== 📦 Checking Installed Packages ==="
pip list | grep -E "pandas|numpy|matplotlib|scikit-learn"
echo

echo "=== 🚀 Running Core Script ==="
python scripts/basic_python.py --in_csv data/day3_sample.csv
echo

echo "=== 📁 Checking Output File ==="
if [ -f reports/figures/day3_event_frequency.png ]; then
    echo "✅ Reproducibility test PASSED: figure found."
else
    echo "❌ Reproducibility test FAILED: output missing."
fi
