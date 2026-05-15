.PHONY: install dev demo demo-replay test test-hook test-cascade test-contract \
        lint format typecheck check bench bench-latency bench-pareto \
        deploy-vultr smoke-vultr clean

# ----- Setup -----

install:
	cd backend && uv sync --extra dev --extra bench
	@mkdir -p ~/.local/bin
	cp sentinel-hook.py ~/.local/bin/sentinel-hook
	chmod +x ~/.local/bin/sentinel-hook
	@echo "Sentinel hook installed at ~/.local/bin/sentinel-hook"
	@echo "Add the following to ~/.claude/settings.json:"
	@echo '  "hooks": { "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "$(HOME)/.local/bin/sentinel-hook"}]}] }'

# ----- Dev loops -----

dev:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 7777

dashboard:
	cd frontend && npm run dev

demo:
	@echo "Day-1 demo: starting daemon. Configure ~/.claude/settings.json hook to point at $(HOME)/.local/bin/sentinel-hook"
	@$(MAKE) dev

demo-replay:
	@echo "Replay mode: feed recorded session traces to the daemon (Day 5)."
	@cd backend && uv run python -m bench.replay --trace data/replay/session-01.jsonl

# ----- Tests -----

test:
	cd backend && uv run pytest

test-hook:
	cd backend && uv run pytest tests/integration/test_hook_subprocess.py -v

test-cascade:
	cd backend && uv run pytest tests/unit/test_layer*.py tests/integration/test_cascade_end_to_end.py -v

test-contract:
	cd backend && uv run pytest tests/contract/ -v

# ----- Quality gates -----

lint:
	cd backend && uv run ruff check .

format:
	cd backend && uv run ruff format .

typecheck:
	cd backend && uv run mypy backend/sentinel

check: lint typecheck
	cd backend && uv run ruff format --check .

# ----- Benchmark -----

bench:
	cd backend && uv run python -m bench.run_bench --dataset ../data/sentinel-bench-v1 --output ../results

bench-latency:
	cd backend && uv run python -m bench.run_bench --latency-only --calls 1000 --gate-ms 10

bench-pareto:
	cd backend && uv run python -m bench.pareto --input ../results/latest.json --output ../results/latest-pareto.png

# ----- Deploy -----

deploy-vultr:
	@test -n "$$SENTINEL_VULTR_HOST" || (echo "Set SENTINEL_VULTR_HOST=user@ip" && exit 1)
	ssh $$SENTINEL_VULTR_HOST "cd /opt/sentinel && git pull && docker-compose -f deploy/docker-compose.yml up -d --build"

smoke-vultr:
	@test -n "$$SENTINEL_DOMAIN" || (echo "Set SENTINEL_DOMAIN=your.domain" && exit 1)
	curl -sSf https://$$SENTINEL_DOMAIN/health | grep -q '"status":"ok"' && echo "Smoke OK"

# ----- Cleanup -----

clean:
	rm -rf backend/.pytest_cache backend/.mypy_cache backend/.ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
