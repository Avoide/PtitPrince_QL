# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-10-03

### Added
- Modern packaging with `pyproject.toml` using Hatchling build backend
- Comprehensive pytest test suite with 60+ tests and 89% code coverage
- Pre-commit hooks with Ruff for code formatting and linting
- Type hints on all public APIs
- CI/CD testing across Python 3.9, 3.10, 3.11, and 3.12
- `MODERNIZATION_NOTES.md` documenting breaking changes

### Changed
- **BREAKING**: Removed default `palette="Set2"` parameter from `RainCloud()` to avoid seaborn 0.14 deprecation warnings
  - Users who want the old behavior should explicitly pass `palette='Set2'`
- Updated GitHub Actions workflow to test multiple Python versions
- Migrated from `setup.py` to modern `pyproject.toml` configuration

### Fixed
- Seaborn 0.13+ compatibility issues
- Dodge alignment in raincloud components
- Pointplot hue handling
- Axis label positioning

## [0.3.0] - Previous Release

### Added
- Seaborn 0.13.2 compatibility
- Updated Jupyter notebooks and figures

### Fixed
- Raincloud component alignment when using hue
- Move offsets with stripplot layering
- Internal refactoring for robustness and clarity

## [0.2.x and earlier]

See git history for changes prior to 0.3.0.

[0.3.1]: https://github.com/pog87/PtitPrince/compare/v0.2.7...v0.3.1
