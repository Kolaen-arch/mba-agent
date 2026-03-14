.PHONY: help install ingest extract serve serve-dev chat status references scaffold import-bib frontend dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt --break-system-packages

ingest: ## Process PDFs into vector store (Claude/RAG path)
	python -m mba_agent ingest

extract: ## Extract cleaned text from PDFs (Gemini/full-context verification)
	python -m mba_agent extract

serve: ## Launch web UI with gunicorn (async, non-blocking)
	gunicorn -k gevent -w 1 -b 0.0.0.0:5000 --timeout 300 "mba_agent.web.app:create_app()"

serve-dev: ## Launch web UI with Flask dev server (auto-reload)
	python -m flask --app "mba_agent.web.app:create_app()" run --host 0.0.0.0 --port 5000 --reload

chat: ## Interactive CLI chat
	python -m mba_agent chat

status: ## Show store info
	python -m mba_agent status

references: ## Show tracked citations
	python -m mba_agent references

scaffold: ## Generate paper_structure.yaml template
	python -m mba_agent scaffold

import-bib: ## Import references from .bib or .json file
	python -m mba_agent import-bib

frontend: ## Build React frontend (output → mba_agent/web/static/dist/)
	cd frontend && npm run build

frontend-dev: ## Run React dev server (Vite + HMR, proxies to Flask)
	cd frontend && npm run dev

dev: ## Run Flask backend + React dev server together
	@echo "Starting Flask on :5000 and Vite on :5173..."
	@echo "Open http://localhost:5173 for development"
	python -m flask --app "mba_agent.web.app:create_app()" run --host 0.0.0.0 --port 5000 --reload &
	cd frontend && npm run dev
