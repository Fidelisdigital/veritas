#!/bin/bash
echo "=================================================="
echo "1. Confirming you're in the right directory"
echo "=================================================="
pwd
ls Makefile pyproject.toml contract/ tests/ 2>&1

echo ""
echo "=================================================="
echo "2. Confirming venv is active and using the right Python"
echo "=================================================="
which python3
python3 --version
echo "^ path above MUST contain '.venv' — if it shows /usr/bin/python3, your venv is NOT active"
echo "  fix: source .venv/bin/activate"

echo ""
echo "=================================================="
echo "3. Confirming all required packages are installed"
echo "=================================================="
python3 -m pip list | grep -Ei "protobuf|fastapi|uvicorn|pydantic|structlog|grpcio|pytest|scikit-learn|pandas|numpy|joblib|requests|python-dotenv"
echo "^ every package above should show a version number. Missing any = incomplete install."

echo ""
echo "=================================================="
echo "4. Confirming proto generation works and files exist"
echo "=================================================="
make proto
ls -la contract/proto/*_pb2.py
echo "^ should list tx_pb2.py, account_pb2.py, etc. with recent timestamps"

echo ""
echo "=================================================="
echo "5. Confirming base tests run (ignore test_contract.py failures — expected until Veritas tx types are added)"
echo "=================================================="
python3 -m pytest tests/test_config.py -v

echo ""
echo "=================================================="
echo "6. Confirming git repo state"
echo "=================================================="
git status
git branch
git log --oneline -5
git remote -v

echo ""
echo "=================================================="
echo "7. Confirming GitHub sync (local vs remote match)"
echo "=================================================="
git fetch origin
git status -uno
echo "^ should say 'Your branch is up to date with origin/main'"

echo ""
echo "=================================================="
echo "DONE — review each section above for red flags"
echo "=================================================="
