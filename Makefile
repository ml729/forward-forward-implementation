.PHONY: requirements

lock:
	poetry lock

export:
	poetry export --dev -f requirements.txt > requirements-dev.txt
	poetry export -f requirements.txt > requirements.txt
