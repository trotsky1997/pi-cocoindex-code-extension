# pi-cocoindex-code-extension

Pi package that bundles two pieces together:

- a `cocoindex-code` powered pi extension that exposes code search as a tool
- a `ccc` skill that teaches the agent when to initialize, index, refresh, and search

The goal is simple: make CocoIndex feel native inside pi without asking the agent to manually manage the full `ccc` lifecycle every time.

## What it includes

- `extensions/cocoindex.ts`
  Registers a `search` tool with parameters aligned to the `cocoindex-code` MCP shape:
  - `query`
  - `limit`
  - `offset`
  - `refresh_index`
  - `languages`
  - `paths`

- `skills/ccc/`
  Bundles the existing `ccc` skill plus reference docs for:
  - installation and management
  - settings and embedding configuration

## Requirements

- [pi](https://github.com/mariozechner/pi-coding-agent)
- `ccc` installed locally

For `ccc` installation, follow the [official CocoIndex Code install guide](https://github.com/cocoindex-io/cocoindex-code?tab=readme-ov-file#install).

Install
Using pipx:

```bash
pipx install cocoindex-code       # first install
pipx upgrade cocoindex-code       # upgrade
```

Using uv:

```bash
uv tool install --upgrade cocoindex-code --prerelease explicit --with "cocoindex>=1.0.0a24"
```

## Install

Install from GitHub:

```bash
pi install git:github.com/trotsky1997/pi-cocoindex-code-extension
```

Or install from a local checkout:

```bash
pi install ./pi-cocoindex-code-extension
```

## Usage

After installing, reload pi or start a new session.

The package gives you:

- a `search` tool for BM25-backed code search
- a `/ccc-status` command
- a `/ccc-patch` command that patches the installed `cocoindex-code` into local BM25 mode after installs/upgrades
- a footer status indicator showing whether CocoIndex is available and initialized
- the `ccc` skill for agent-side search workflow guidance

Typical usage inside pi:

```text
search authentication flow
search trainer initialization logic
search error handling retry logic
```

The extension will automatically:

- locate the project root
- patch `cocoindex-code` into local BM25 mode, switch `~/.cocoindex_code/global_settings.yml` to `provider: bm25`, and keep the BM25 FTS cache in `target_sqlite.db`
- detect whether `.cocoindex_code/settings.yml` exists
- run `ccc init` and `ccc index` on first search if needed
- refresh footer status on new turns

## Notes

- The extension currently wraps the local `ccc` CLI rather than embedding `ccc mcp` directly.
- The `search` tool shape is intentionally kept close to the CocoIndex MCP interface.
- The auto-patch is idempotent: it rewrites known `cocoindex-code` Python blocks, enables a cached SQLite FTS5/BM25 index, and preserves existing `envs` in `global_settings.yml`. If `ccc` is upgraded, use `/ccc-patch` to retry and inspect the result.
- BM25 ranking runs inside SQLite FTS5; query normalization is cached in-process, the extension now attempts to install `rapidfuzz` into the `ccc` Python environment for fuzzy reranking, and NumPy is only used for a tiny compatibility vector with reused batch allocations so indexing stays local and cheap.
- If you already have a global `ccc` skill installed, pi may prefer the first discovered skill with that name.

## Repository

- GitHub: `https://github.com/trotsky1997/pi-cocoindex-code-extension`
