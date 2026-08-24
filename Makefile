VENV_DIR := $(abspath .venv-lint)
PIP := $(VENV_DIR)/bin/pip
PYTHON := $(VENV_DIR)/bin/python
RUFF := $(VENV_DIR)/bin/ruff
DPRINT := $(VENV_DIR)/bin/dprint
TARGET_DIRS ?= .
CHECK ?= 0

.PHONY: clean
clean:
	git clean -Xdf --exclude "!.env"

.PHONY: setup-venv
setup-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment for linters..."; \
		python3 -m venv $(VENV_DIR); \
		$(PIP) install --upgrade pip; \
		$(PIP) install ruff; \
	fi

.PHONY: setup-dprint
setup-dprint: setup-venv
	@if [ ! -f "$(DPRINT)" ]; then \
		echo "Installing dprint to $(VENV_DIR)/bin..."; \
		mkdir -p $(VENV_DIR)/bin; \
		curl -fsSL https://dprint.dev/install.sh | DPRINT_INSTALL=$(VENV_DIR) sh; \
	fi

.PHONY: setup-env
setup-env: setup-venv setup-dprint

.PHONY: ruff
ruff: setup-venv
	@if [ "$(CHECK)" = "1" ]; then \
		echo "Running Ruff in check mode..."; \
		$(RUFF) check $(TARGET_DIRS); \
		LINT_STATUS=$$?; \
		$(RUFF) format $(TARGET_DIRS) --check; \
		FMT_STATUS=$$?; \
		if [ $$LINT_STATUS -ne 0 ] || [ $$FMT_STATUS -ne 0 ]; then exit 1; fi; \
	else \
		echo "Running Ruff and fixing issues..."; \
		$(RUFF) check $(TARGET_DIRS) --fix; \
		LINT_STATUS=$$?; \
		$(RUFF) format $(TARGET_DIRS); \
		FMT_STATUS=$$?; \
		if [ $$LINT_STATUS -ne 0 ] || [ $$FMT_STATUS -ne 0 ]; then exit 1; fi; \
	fi

.PHONY: dprint
dprint: setup-dprint
	@if [ "$(CHECK)" = "1" ]; then \
		echo "Running dprint in check mode..."; \
		$(DPRINT) check $(TARGET_DIRS); \
	else \
		echo "Running dprint and formatting..."; \
		$(DPRINT) fmt $(TARGET_DIRS); \
	fi

.PHONY: lint
lint: ruff dprint
