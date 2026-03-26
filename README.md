# pi-cocoindex-code-extension

Pi package that bundles two pieces together:

- a `cocoindex-code` powered pi extension that exposes semantic code search as a tool
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

Install `ccc` with one of the supported methods from `cocoindex-code`, for example:

```bash
pipx install cocoindex-code
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

- a `search` tool for semantic code search
- a `/ccc-status` command
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
- detect whether `.cocoindex_code/settings.yml` exists
- run `ccc init` and `ccc index` on first search if needed
- refresh footer status on new turns

## Notes

- The extension currently wraps the local `ccc` CLI rather than embedding `ccc mcp` directly.
- The `search` tool shape is intentionally kept close to the CocoIndex MCP interface.
- If you already have a global `ccc` skill installed, pi may prefer the first discovered skill with that name.

## Repository

- GitHub: `https://github.com/trotsky1997/pi-cocoindex-code-extension`
