.PHONY: help install ingest serve serve-dev chat status references scaffold import-bib

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt --break-system-packages

ingest: ## Process PDFs into vector store
	python -m mba_agent ingest

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
