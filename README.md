# pi-cocoindex-code-extension

Pi package bundling:

- a local CocoIndex-powered `search` extension for pi
- the `ccc` skill for semantic code search lifecycle guidance

## Install locally

```bash
pi install ./pi-cocoindex-code-extension
```

Or load it for one run:

```bash
pi -e ./pi-cocoindex-code-extension/extensions/cocoindex.ts
```

## Contents

- `extensions/cocoindex.ts` — registers the `search` tool, `/ccc-status`, and a footer status indicator
- `skills/ccc/` — loads the `ccc` skill plus management/settings references

## Notes

- The extension uses the local `ccc` CLI and auto-runs `ccc init` + `ccc index` on first search when needed.
- The skill teaches the agent when to use semantic search and how to manage index freshness.
