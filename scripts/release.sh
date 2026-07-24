#!/usr/bin/env bash
# QbitOS Local Release Helper
#
# Usage:
#   ./scripts/release.sh <version>      # e.g. ./scripts/release.sh 0.2.0
#
# What this script does:
#   1. Validates the provided version (semver) and ensures no tag exists yet.
#   2. Bumps the workspace version in the root Cargo.toml using sed.
#   3. Builds release binaries for 4 targets:
#        - x86_64-unknown-linux-gnu
#        - x86_64-pc-windows-msvc
#        - x86_64-apple-darwin
#        - aarch64-apple-darwin
#      (Cross-compilation targets must be pre-installed via `rustup target add`.)
#   4. Packages each binary with migrations/README/LICENSE/.env.example.
#   5. Generates SHA256SUMS for all artifacts.
#   6. Optionally signs SHA256SUMS with GPG (set QBITOS_GPG_KEY_ID env var).
#   7. Creates an annotated git tag `v<version>`.
#
# Environment variables:
#   QBITOS_GPG_KEY_ID   - GPG key ID for signing SHA256SUMS (optional)
#   QBITOS_SKIP_BUILD   - if set to "1", skip the cargo build step
#   QBITOS_TARGETS      - space-separated list of targets to build (defaults to all 4)

set -euo pipefail

# ─── Helpers ─────────────────────────────────────────────────────────────────
RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
BLUE=$'\033[34m'
RESET=$'\033[0m'

log()  { printf "${BLUE}▶ %s${RESET}\n" "$1"; }
ok()   { printf "${GREEN}✓ %s${RESET}\n" "$1"; }
warn() { printf "${YELLOW}⚠ %s${RESET}\n" "$1"; }
die()  { printf "${RED}✗ %s${RESET}\n" "$1" >&2; exit 1; }

# ─── 0. Argument validation ──────────────────────────────────────────────────
if [[ $# -ne 1 ]]; then
    die "Usage: $0 <version>   (e.g. $0 0.2.0)"
fi

VERSION="$1"
if ! [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    die "Version '${VERSION}' is not a valid semver (expected X.Y.Z or X.Y.Z-pre)"
fi

TAG="v${VERSION}"

# Resolve repo root (this script lives in scripts/).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# ─── 1. Pre-flight checks ────────────────────────────────────────────────────
log "Pre-flight checks"
if ! command -v cargo > /dev/null 2>&1; then
    die "cargo not found on PATH. Install Rust 1.85+ and try again."
fi
if ! command -v git > /dev/null 2>&1; then
    die "git not found on PATH."
fi
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    die "Not inside a git repository: ${REPO_ROOT}"
fi
# Require a clean working tree (allow untracked files only if explicitly permitted).
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    die "Working tree is dirty. Commit or stash changes before releasing."
fi
# Check tag doesn't already exist.
if git rev-parse "${TAG}" > /dev/null 2>&1; then
    die "Git tag '${TAG}' already exists. Pick a different version or delete the tag."
fi
ok "Pre-flight checks passed"

# ─── 2. Bump workspace version ───────────────────────────────────────────────
log "Bumping workspace version to ${VERSION}"
CARGO_TOML="${REPO_ROOT}/Cargo.toml"
if ! grep -q '^\[workspace\.package\]' "${CARGO_TOML}"; then
    die "Could not find [workspace.package] section in ${CARGO_TOML}"
fi
# Use sed to replace `version = "X.Y.Z"` only inside the [workspace.package] section.
# We do this by limiting the edit to lines after the [workspace.package] marker.
python3 - <<PY || die "Failed to bump version in ${CARGO_TOML}"
import re, sys
path = "${CARGO_TOML}"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
# Match the [workspace.package] section, then the first version= line within it.
pattern = re.compile(
    r'(\[workspace\.package\][^\[]*?version\s*=\s*)"([^"]+)"',
    re.DOTALL,
)
new_content, n = pattern.subn(r'\g<1>"${VERSION}"', content, count=1)
if n != 1:
    print("ERROR: did not find version line under [workspace.package]", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8") as f:
    f.write(new_content)
print(f"Bumped version to ${VERSION}")
PY
ok "Workspace version bumped to ${VERSION}"

# Update Cargo.lock so the workspace metadata is consistent.
log "Updating Cargo.lock"
cargo generate-lockfile --offline 2>/dev/null || cargo generate-lockfile
ok "Cargo.lock updated"

# ─── 3. Build release binaries ───────────────────────────────────────────────
DEFAULT_TARGETS=(
    "x86_64-unknown-linux-gnu"
    "x86_64-pc-windows-msvc"
    "x86_64-apple-darwin"
    "aarch64-apple-darwin"
)
read -ra TARGETS <<< "${QBITOS_TARGETS:-${DEFAULT_TARGETS[*]}}"

STAGE_DIR="${REPO_ROOT}/target/release-artifacts"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

if [[ "${QBITOS_SKIP_BUILD:-0}" == "1" ]]; then
    warn "QBITOS_SKIP_BUILD=1 — skipping cargo build step"
else
    for target in "${TARGETS[@]}"; do
        log "Building release for ${target}"
        # Ensure target is installed (no-op if already present).
        rustup target add "${target}" 2>/dev/null || warn "rustup target add ${target} failed (continuing)"
        if ! cargo build --release -p qbitos-cli --target "${target}"; then
            die "cargo build failed for target ${target}"
        fi
        ok "Built ${target}"
    done
fi

# ─── 4. Package each target ──────────────────────────────────────────────────
log "Packaging release artifacts"
ARTIFACT_LIST=()
for target in "${TARGETS[@]}"; do
    STAGE="${STAGE_DIR}/qbitos-${VERSION}-${target}"
    mkdir -p "${STAGE}/memory/migrations"

    # Pick the binary name (qbitos vs qbitos.exe).
    if [[ "${target}" == *"-windows-"* ]]; then
        BIN_NAME="qbitos.exe"
    else
        BIN_NAME="qbitos"
    fi
    BIN_PATH="${REPO_ROOT}/target/${target}/release/${BIN_NAME}"
    if [[ ! -f "${BIN_PATH}" ]]; then
        die "Release binary not found: ${BIN_PATH}"
    fi
    cp "${BIN_PATH}" "${STAGE}/${BIN_NAME}"

    # Bundle resources.
    if [[ -d "${REPO_ROOT}/memory/migrations" ]]; then
        cp -r "${REPO_ROOT}/memory/migrations/." "${STAGE}/memory/migrations/"
    fi
    [[ -f "${REPO_ROOT}/README.md" ]] && cp "${REPO_ROOT}/README.md" "${STAGE}/"
    [[ -f "${REPO_ROOT}/LICENSE" ]] && cp "${REPO_ROOT}/LICENSE" "${STAGE}/"
    [[ -f "${REPO_ROOT}/.env.example" ]] && cp "${REPO_ROOT}/.env.example" "${STAGE}/" || true

    # Create archive (tar.gz for unix, zip for windows).
    if [[ "${target}" == *"-windows-"* ]]; then
        ARTIFACT="${STAGE_DIR}/qbitos-${VERSION}-${target}.zip"
        ( cd "${STAGE_DIR}" && zip -r -q "$(basename "${ARTIFACT}")" "qbitos-${VERSION}-${target}" )
    else
        ARTIFACT="${STAGE_DIR}/qbitos-${VERSION}-${target}.tar.gz"
        tar -czf "${ARTIFACT}" -C "${STAGE_DIR}" "qbitos-${VERSION}-${target}"
    fi
    ARTIFACT_LIST+=("${ARTIFACT}")
    ok "Packaged $(basename "${ARTIFACT}")"
done

# ─── 5. Generate SHA256SUMS ──────────────────────────────────────────────────
log "Generating SHA256SUMS"
SHA256_FILE="${STAGE_DIR}/SHA256SUMS"
: > "${SHA256_FILE}"
for artifact in "${ARTIFACT_LIST[@]}"; do
    base="$(basename "${artifact}")"
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "${artifact}" | sed "s|${STAGE_DIR}/||" >> "${SHA256_FILE}"
    else
        shasum -a 256 "${artifact}" | sed "s|${STAGE_DIR}/||" >> "${SHA256_FILE}"
    fi
done
ok "SHA256SUMS written to ${SHA256_FILE}"
cat "${SHA256_FILE}"

# ─── 6. GPG sign (optional) ──────────────────────────────────────────────────
if [[ -n "${QBITOS_GPG_KEY_ID:-}" ]]; then
    log "Signing SHA256SUMS with GPG key ${QBITOS_GPG_KEY_ID}"
    if ! command -v gpg > /dev/null 2>&1; then
        die "gpg not found on PATH but QBITOS_GPG_KEY_ID is set"
    fi
    gpg --batch --yes --local-user "${QBITOS_GPG_KEY_ID}" \
        --output "${SHA256_FILE}.sig" \
        --detach-sign "${SHA256_FILE}"
    ok "Signature written to ${SHA256_FILE}.sig"
else
    warn "QBITOS_GPG_KEY_ID not set — skipping GPG signature"
fi

# ─── 7. Create git tag ───────────────────────────────────────────────────────
log "Creating git tag ${TAG}"
git add "${CARGO_TOML}" "${REPO_ROOT}/Cargo.lock"
git commit -m "chore: release ${TAG}" 2>/dev/null || warn "Nothing to commit (Cargo.toml unchanged?)"
git tag -a "${TAG}" -m "QbitOS release ${TAG}

Artifacts:
$(printf '  - %s\n' "${ARTIFACT_LIST[@]}")

SHA256SUMS:
$(cat "${SHA256_FILE}")
"
ok "Created annotated tag ${TAG}"

# ─── Summary ─────────────────────────────────────────────────────────────────
printf "\n${GREEN}=== Release ${TAG} prepared ===${RESET}\n"
printf "Artifacts in: ${STAGE_DIR}\n"
printf "Tag:          ${TAG}\n"
printf "Push with:    git push origin main ${TAG}\n"
