THIS_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
BIN_DIR := $(THIS_DIR)/bin
TMP_DIR := $(THIS_DIR)/tmp

# Allow override via environment: RG_VERSION=15.0.0 make download-ripgrep
RG_VERSION ?= 15.0.0
OS := $(shell uname -s)
ARCH := $(shell uname -m | tr '[:upper:]' '[:lower:]')
RG_ARCHIVE_EXT := tar.gz
RG_ARCHIVE_BIN := rg
RG_BIN_SUFFIX :=

# Map OS/ARCH to ripgrep TARGET name
# See: https://github.com/BurntSushi/ripgrep/releases
ifeq ($(OS),Darwin)
  ifeq ($(ARCH),arm64)
    RG_TARGET := aarch64-apple-darwin
  else ifeq ($(ARCH),x86_64)
    RG_TARGET := x86_64-apple-darwin
  else
    $(error Unsupported macOS architecture: $(ARCH))
  endif
else ifeq ($(OS),Linux)
  ifeq ($(ARCH),x86_64)
    RG_TARGET := x86_64-unknown-linux-musl
  else ifeq ($(ARCH),aarch64)
    RG_TARGET := aarch64-unknown-linux-gnu
  else ifeq ($(ARCH),armv7l)
    RG_TARGET := arm-unknown-linux-gnueabihf
  else
    $(error Unsupported Linux architecture: $(ARCH))
  endif
else ifneq (,$(filter MSYS% MINGW%,$(OS)))
  ifeq ($(ARCH),x86_64)
    RG_TARGET := x86_64-pc-windows-msvc
  else ifeq ($(ARCH),aarch64)
    RG_TARGET := aarch64-pc-windows-msvc
  else
    $(error Unsupported Windows architecture: $(ARCH))
  endif
  RG_ARCHIVE_EXT := zip
  RG_ARCHIVE_BIN := rg.exe
  RG_BIN_SUFFIX := .exe
else
  $(error Unsupported OS: $(OS))
endif

RG_ARCHIVE := ripgrep-$(RG_VERSION)-$(RG_TARGET).$(RG_ARCHIVE_EXT)
RG_URL := https://github.com/BurntSushi/ripgrep/releases/download/$(RG_VERSION)/$(RG_ARCHIVE)


.PHONY: download-ripgrep
download-ripgrep:
	@echo "==> Ensuring ripgrep is installed"
	@if [ -f "$(BIN_DIR)/rg$(RG_BIN_SUFFIX)" ]; then \
		echo "rg already installed at $(BIN_DIR)/rg$(RG_BIN_SUFFIX)"; \
	else \
		echo "Downloading ripgrep $(RG_VERSION) from: $(RG_URL)"; \
		mkdir -p "$(BIN_DIR)" "$(TMP_DIR)"; \
		ARCHIVE_PATH="$(TMP_DIR)/$(RG_ARCHIVE)"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L --fail -o "$$ARCHIVE_PATH" "$(RG_URL)"; \
		else \
			if command -v wget >/dev/null 2>&1; then \
				wget -O "$$ARCHIVE_PATH" "$(RG_URL)"; \
			else \
				echo "Error: neither curl nor wget is available" && exit 1; \
			fi; \
		fi; \
		if [ "$(RG_ARCHIVE_EXT)" = "zip" ]; then \
			ARCHIVE_PATH="$$ARCHIVE_PATH" TMP_DIR="$(TMP_DIR)" python -c "import os, zipfile; zipfile.ZipFile(os.environ['ARCHIVE_PATH']).extractall(os.environ['TMP_DIR'])"; \
		else \
			tar -xzf "$$ARCHIVE_PATH" -C "$(TMP_DIR)"; \
		fi; \
		SRC_PATH="$(TMP_DIR)/ripgrep-$(RG_VERSION)-$(RG_TARGET)/$(RG_ARCHIVE_BIN)"; \
		DST_PATH="$(BIN_DIR)/rg$(RG_BIN_SUFFIX)"; \
		cp "$$SRC_PATH" "$$DST_PATH"; \
		chmod +x "$$DST_PATH"; \
		echo "rg installed at $$DST_PATH"; \
	fi


# Portable shell shipped for Windows machines without Git for Windows.
#
# Installed as `sh.exe` on purpose: kimi-cli invokes the shell as
# `<path> -c <command>`, and busybox dispatches on argv[0] — a copy still named
# busybox.exe would look for an applet called "-c".
#
# Unlike ripgrep this is fetched for the *target*, not the build host: the
# Windows installer can be staged from macOS or Linux, and the binary has to be
# in the source tree either way. Default target matches the installer's
# x86_64 build (packaging/build_windows.py); pass BUSYBOX_ARCH=arm64 for
# Windows-on-ARM.
#
# Both defaults are the Unicode builds: without them busybox mangles non-ASCII
# paths and output, which is not optional for CJK users. They need Windows 10
# 1903 or newer — older hosts still have git-bash and KIMI_CLI_SHELL_PATH.
#
# Checksums are upstream's, from https://frippery.org/files/busybox/SHA256SUM
# (PGP-signed as SHA256SUM.sig). Verify the signature before bumping the pin.
BUSYBOX_VERSION ?= FRP-6075-g169694ebd
BUSYBOX_ARCH ?= x86_64
ifeq ($(BUSYBOX_ARCH),x86_64)
  BUSYBOX_FILE := busybox-w64u-$(BUSYBOX_VERSION).exe
  BUSYBOX_SHA256 := 6e263d154d8548d1eb936f65d1d8312c80df31c45974e48d6335e4dcc0f4f34c
else ifeq ($(BUSYBOX_ARCH),arm64)
  BUSYBOX_FILE := busybox-w64a-$(BUSYBOX_VERSION).exe
  BUSYBOX_SHA256 := e67f873d19d58c535cc9f0c4965ffd622e19b7bab87e3da89cb2185fb54464d7
else
  $(error Unsupported BUSYBOX_ARCH: $(BUSYBOX_ARCH) (expected x86_64 or arm64))
endif
BUSYBOX_URL ?= https://frippery.org/files/busybox/$(BUSYBOX_FILE)

.PHONY: download-busybox
download-busybox:
	@echo "==> Ensuring portable shell (busybox) is installed"
	@if [ -f "$(BIN_DIR)/sh.exe" ]; then \
		echo "sh.exe already installed at $(BIN_DIR)/sh.exe"; \
	else \
		echo "Downloading busybox $(BUSYBOX_VERSION) ($(BUSYBOX_ARCH)) from: $(BUSYBOX_URL)"; \
		mkdir -p "$(BIN_DIR)" "$(TMP_DIR)"; \
		DOWNLOAD_PATH="$(TMP_DIR)/$(BUSYBOX_FILE)"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L --fail -o "$$DOWNLOAD_PATH" "$(BUSYBOX_URL)"; \
		else \
			if command -v wget >/dev/null 2>&1; then \
				wget -O "$$DOWNLOAD_PATH" "$(BUSYBOX_URL)"; \
			else \
				echo "Error: neither curl nor wget is available" && exit 1; \
			fi; \
		fi; \
		DOWNLOAD_PATH="$$DOWNLOAD_PATH" EXPECTED="$(BUSYBOX_SHA256)" python3 -c "import hashlib, os, sys; d=hashlib.sha256(open(os.environ['DOWNLOAD_PATH'],'rb').read()).hexdigest(); sys.exit(0) if d==os.environ['EXPECTED'] else sys.exit('sha256 mismatch: got %s' % d)"; \
		cp "$$DOWNLOAD_PATH" "$(BIN_DIR)/sh.exe"; \
		chmod +x "$(BIN_DIR)/sh.exe"; \
		echo "sh.exe installed at $(BIN_DIR)/sh.exe"; \
	fi


.PHONY: download-deps
download-deps: download-ripgrep download-busybox
