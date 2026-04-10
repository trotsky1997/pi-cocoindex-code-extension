import { spawn } from "node:child_process";
import { access, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const STATUS_KEY = "cocoindex";

type StatusState = {
	available: boolean;
	busy: false | "checking" | "indexing" | "patching" | "searching";
	initialized: boolean;
	projectRoot: string;
	lastSummary: string;
};

type SearchParams = {
	query: string;
	limit?: number;
	offset?: number;
	refresh_index?: boolean;
	languages?: string[];
	paths?: string[];
};

type SearchResult = {
	file_path: string;
	language: string;
	content: string;
	start_line: number;
	end_line: number;
	score: number;
};

type CommandResult = {
	code: number;
	stdout: string;
	stderr: string;
	combined: string;
};

type PatchResult = {
	status: "already_patched" | "failed" | "missing" | "patched" | "skipped";
	message: string;
	targetPath?: string;
};

const SEARCH_PARAMS = Type.Object({
	query: Type.String({ description: "Natural language query or code snippet to search for." }),
	limit: Type.Optional(
		Type.Integer({ minimum: 1, maximum: 100, default: 5, description: "Maximum number of results to return (1-100)" }),
	),
	offset: Type.Optional(
		Type.Integer({ minimum: 0, default: 0, description: "Number of results to skip for pagination" }),
	),
	refresh_index: Type.Optional(
		Type.Boolean({ default: true, description: "Whether to incrementally refresh the index before searching" }),
	),
	languages: Type.Optional(
		Type.Array(Type.String(), { description: "Filter by language(s), e.g. ['python', 'typescript']" }),
	),
	paths: Type.Optional(
		Type.Array(Type.String(), { description: "Filter by file path pattern(s) using glob wildcards" }),
	),
});

const COMPAT_PATCH_MARKER = "null encoding_format, so force the standard float response here.";
const SHARED_IMPORT_NEEDLE = "from dataclasses import dataclass\n";
const VULNERABLE_LITELLM_BLOCK = `    else:
        from cocoindex.ops.litellm import LiteLLMEmbedder

        instance = LiteLLMEmbedder(settings.model)
        query_prompt_name = None
        logger.info("Embedding model (LiteLLM): %s", settings.model)
`;
const PATCHED_LITELLM_BLOCK = `    else:
        from cocoindex.ops.litellm import LiteLLMEmbedder

        litellm_kwargs = {}
        # Some OpenAI-compatible embedding APIs reject LiteLLM's default
        # null encoding_format, so force the standard float response here.
        if settings.model.startswith("openai/") or os.environ.get("OPENAI_API_BASE"):
            litellm_kwargs["encoding_format"] = "float"

        instance = LiteLLMEmbedder(settings.model, **litellm_kwargs)
        query_prompt_name = None
        logger.info("Embedding model (LiteLLM): %s", settings.model)
`;

let patchPromise: Promise<PatchResult> | null = null;
let patchResult: PatchResult | null = null;

function createDefaultState(): StatusState {
	return {
		available: true,
		busy: false,
		initialized: false,
		projectRoot: process.cwd(),
		lastSummary: "idle",
	};
}

function isInitError(text: string): boolean {
	return text.includes("Run `ccc init`") || text.includes("Not in an initialized project directory") || text.includes("Global settings not found");
}

async function pathExists(target: string): Promise<boolean> {
	try {
		await access(target);
		return true;
	} catch {
		return false;
	}
}

function isCompatParameterError(text: string): boolean {
	return text.includes("code': 20015") || text.includes('"code":20015') || text.includes("The parameter is invalid. Please check again.");
}

function parseShebangInterpreter(scriptText: string): string | null {
	const [firstLine = ""] = scriptText.split(/\r?\n/, 1);
	if (!firstLine.startsWith("#!")) return null;
	const shebang = firstLine.slice(2).trim();
	if (!shebang) return null;
	const parts = shebang.split(/\s+/);
	if (parts[0].endsWith("env") && parts[1]) {
		return parts[1];
	}
	return parts[0] ?? null;
}

async function findCommandPath(command: string): Promise<string | null> {
	if (command.includes(path.sep)) {
		return (await pathExists(command)) ? command : null;
	}

	const pathValue = process.env.PATH ?? "";
	for (const dir of pathValue.split(path.delimiter)) {
		if (!dir) continue;
		const candidate = path.join(dir, command);
		if (await pathExists(candidate)) {
			return candidate;
		}
	}

	return null;
}

async function findProjectRoot(startDir: string): Promise<string> {
	let current = path.resolve(startDir);
	while (true) {
		if (await pathExists(path.join(current, ".cocoindex_code", "settings.yml"))) {
			return current;
		}
		if (await pathExists(path.join(current, ".git"))) {
			return current;
		}
		const parent = path.dirname(current);
		if (parent === current) {
			return path.resolve(startDir);
		}
		current = parent;
	}
}

function renderStatus(ctx: ExtensionContext, state: StatusState): void {
	const theme = ctx.ui.theme;
	if (!state.available) {
		ctx.ui.setStatus(STATUS_KEY, theme.fg("warning", "ccc missing"));
		return;
	}

	if (state.busy) {
		const dot = theme.fg("accent", "●");
		ctx.ui.setStatus(STATUS_KEY, `${dot}${theme.fg("dim", ` ccc ${state.busy}...`)}`);
		return;
	}

	if (!state.initialized) {
		ctx.ui.setStatus(STATUS_KEY, theme.fg("warning", "ccc not initialized"));
		return;
	}

	const check = theme.fg("success", "✓");
	const rootName = path.basename(state.projectRoot) || state.projectRoot;
	ctx.ui.setStatus(STATUS_KEY, `${check}${theme.fg("dim", ` ccc ${rootName} · ${state.lastSummary}`)}`);
}

function runCommand(command: string, args: string[], cwd: string, signal?: AbortSignal): Promise<CommandResult> {
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, {
			cwd,
			env: process.env,
			stdio: ["ignore", "pipe", "pipe"],
		});

		let stdout = "";
		let stderr = "";

		child.stdout.on("data", (chunk: Buffer | string) => {
			stdout += chunk.toString();
		});
		child.stderr.on("data", (chunk: Buffer | string) => {
			stderr += chunk.toString();
		});

		const onAbort = () => {
			child.kill("SIGTERM");
			reject(new Error("Command aborted"));
		};

		signal?.addEventListener("abort", onAbort, { once: true });

		child.on("error", (error) => {
			signal?.removeEventListener("abort", onAbort);
			reject(error);
		});

		child.on("close", (code) => {
			signal?.removeEventListener("abort", onAbort);
			resolve({
				code: code ?? 1,
				stdout,
				stderr,
				combined: `${stdout}${stderr}`.trim(),
			});
		});
	});
}

function buildCompatibilityPatchedSource(source: string): string | null {
	if (source.includes(COMPAT_PATCH_MARKER) || source.includes('litellm_kwargs["encoding_format"] = "float"')) {
		return source;
	}

	let updated = source;
	if (!updated.includes("\nimport os\n")) {
		if (!updated.includes(SHARED_IMPORT_NEEDLE)) {
			return null;
		}
		updated = updated.replace(SHARED_IMPORT_NEEDLE, `${SHARED_IMPORT_NEEDLE}import os\n`);
	}

	if (!updated.includes(VULNERABLE_LITELLM_BLOCK)) {
		return null;
	}

	return updated.replace(VULNERABLE_LITELLM_BLOCK, PATCHED_LITELLM_BLOCK);
}

async function locateSharedModulePath(projectRoot: string): Promise<string> {
	const cccPath = await findCommandPath("ccc");
	if (!cccPath) {
		throw new Error("ccc not found on PATH");
	}

	const launcherText = await readFile(cccPath, "utf8");
	const pythonHint = parseShebangInterpreter(launcherText);
	if (!pythonHint) {
		throw new Error(`Could not parse the ccc launcher at ${cccPath}`);
	}

	const pythonPath = pythonHint.includes(path.sep) ? pythonHint : (await findCommandPath(pythonHint)) ?? pythonHint;
	const locateResult = await runCommand(
		pythonPath,
		["-c", "import cocoindex_code.shared as module; print(module.__file__)"],
		projectRoot,
	);
	if (locateResult.code !== 0) {
		throw new Error(locateResult.combined || "Failed to locate cocoindex_code.shared");
	}

	const sharedPath = locateResult.stdout
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean);
	const sharedPathResult = sharedPath[sharedPath.length - 1];
	if (!sharedPathResult) {
		throw new Error("cocoindex_code.shared returned an empty path");
	}

	return sharedPathResult;
}

async function applyCompatibilityPatch(projectRoot: string): Promise<PatchResult> {
	const cccPath = await findCommandPath("ccc");
	if (!cccPath) {
		return { status: "missing", message: "ccc not found on PATH" };
	}

	try {
		const sharedPath = await locateSharedModulePath(projectRoot);
		const source = await readFile(sharedPath, "utf8");
		if (source.includes(COMPAT_PATCH_MARKER) || source.includes('litellm_kwargs["encoding_format"] = "float"')) {
			return {
				status: "already_patched",
				message: `Compatibility patch already present at ${sharedPath}`,
				targetPath: sharedPath,
			};
		}

		const patchedSource = buildCompatibilityPatchedSource(source);
		if (!patchedSource || patchedSource === source) {
			return {
				status: "skipped",
				message: `Unsupported cocoindex_code.shared.py layout at ${sharedPath}; compatibility patch not applied.`,
				targetPath: sharedPath,
			};
		}

		await writeFile(sharedPath, patchedSource, "utf8");
		const restartResult = await runCommand("ccc", ["daemon", "restart"], projectRoot);
		const restartMessage =
			restartResult.code === 0
				? " and restarted the ccc daemon."
				: `, but restarting the ccc daemon failed: ${restartResult.combined || "unknown error"}`;
		return {
			status: "patched",
			message: `Patched ${sharedPath}${restartMessage}`,
			targetPath: sharedPath,
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		return { status: "failed", message: `Compatibility patch failed: ${message}` };
	}
}

async function ensureCompatibilityPatch(projectRoot: string, options: { force?: boolean } = {}): Promise<PatchResult> {
	if (patchPromise) {
		return patchPromise;
	}
	if (!options.force && patchResult) {
		return patchResult;
	}

	patchPromise = applyCompatibilityPatch(projectRoot)
		.then((result) => {
			patchResult = result;
			return result;
		})
		.finally(() => {
			patchPromise = null;
		});

	return patchPromise;
}

function buildCompatFailureMessage(message: string, compatResult: PatchResult | null): string {
	if (!isCompatParameterError(message)) {
		return message;
	}

	const patchSummary = compatResult ? compatResult.message : "Compatibility patch status is unavailable.";
	const retryHint =
		compatResult?.status === "patched" || compatResult?.status === "already_patched"
			? " If the old daemon is still serving requests, rerun `/ccc-patch` or retry once."
			: " Run `/ccc-patch` to retry the self-heal patch.";
	return `Detected the known LiteLLM/OpenAI-compatible embedding error from cocoindex-code. ${patchSummary}${retryHint}`;
}

function parseSearchResults(stdout: string): SearchResult[] {
	const matches = [...stdout.matchAll(/--- Result \d+ \(score: ([0-9.]+)\) ---\nFile: (.+?):(\d+)-(\d+) \[([^\]]+)\]\n([\s\S]*?)(?=\n--- Result \d+ \(score:|$)/g)];
	return matches.map((match) => ({
		score: Number(match[1]),
		file_path: match[2],
		start_line: Number(match[3]),
		end_line: Number(match[4]),
		language: match[5],
		content: match[6].trimEnd(),
	}));
}

async function ensureSearchReady(
	projectRoot: string,
	signal: AbortSignal | undefined,
	onUpdate: (message: string) => void,
): Promise<void> {
	onUpdate("Initializing cocoindex project...");
	const initResult = await runCommand("ccc", ["init"], projectRoot, signal);
	if (initResult.code !== 0) {
		throw new Error(initResult.combined || "ccc init failed");
	}

	onUpdate("Building cocoindex index...");
	const indexResult = await runCommand("ccc", ["index"], projectRoot, signal);
	if (indexResult.code !== 0) {
		throw new Error(indexResult.combined || "ccc index failed");
	}
}

async function refreshState(ctx: ExtensionContext, state: StatusState): Promise<void> {
	state.projectRoot = await findProjectRoot(process.cwd());
	const settingsPath = path.join(state.projectRoot, ".cocoindex_code", "settings.yml");
	try {
		state.available = true;
		state.initialized = await pathExists(settingsPath);
		state.lastSummary = state.initialized ? "ready" : "needs init";
	} catch {
		state.available = false;
		state.initialized = false;
		state.lastSummary = "unavailable";
	}
	renderStatus(ctx, state);
}

export default function cocoindexExtension(pi: ExtensionAPI) {
	const state = createDefaultState();

	const notifyPatchResult = (ctx: ExtensionContext, result: PatchResult, manual = false) => {
		if (result.status === "patched") {
			ctx.ui.notify(`ccc patch: ${result.message}`, "info");
			return;
		}
		if (manual) {
			ctx.ui.notify(`ccc patch: ${result.message}`, result.status === "failed" || result.status === "missing" || result.status === "skipped" ? "warning" : "info");
			return;
		}
		if (result.status === "failed" || result.status === "skipped") {
			ctx.ui.notify(`ccc patch: ${result.message}`, "warning");
		}
	};

	const runPatchPreflight = async (ctx: ExtensionContext, options: { force?: boolean; manual?: boolean } = {}) => {
		state.projectRoot = await findProjectRoot(process.cwd());
		const previousBusy = state.busy;
		state.busy = "patching";
		renderStatus(ctx, state);
		const result = await ensureCompatibilityPatch(state.projectRoot, { force: options.force });
		if (state.busy === "patching") {
			state.busy = previousBusy === "patching" ? false : previousBusy;
		}
		renderStatus(ctx, state);
		notifyPatchResult(ctx, result, options.manual);
		return result;
	};

	const kickOffPatchPreflight = (ctx: ExtensionContext) => {
		if (patchPromise || patchResult) return;
		void runPatchPreflight(ctx).catch((error) => {
			if (state.busy === "patching") {
				state.busy = false;
			}
			renderStatus(ctx, state);
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`ccc patch: ${message}`, "warning");
		});
	};

	pi.on("session_start", async (_event, ctx) => {
		state.busy = "checking";
		renderStatus(ctx, state);
		await refreshState(ctx, state);
		state.busy = false;
		renderStatus(ctx, state);
		kickOffPatchPreflight(ctx);
	});

	pi.on("session_switch", async (event, ctx) => {
		if (event.reason !== "new") return;
		state.busy = "checking";
		renderStatus(ctx, state);
		await refreshState(ctx, state);
		state.busy = false;
		renderStatus(ctx, state);
		kickOffPatchPreflight(ctx);
	});

	pi.on("turn_start", async (_event, ctx) => {
		if (state.busy) return;
		await refreshState(ctx, state);
	});

	pi.registerCommand("ccc-status", {
		description: "Refresh the cocoindex status indicator",
		handler: async (_args, ctx) => {
			state.busy = "checking";
			renderStatus(ctx, state);
			await refreshState(ctx, state);
			state.busy = false;
			renderStatus(ctx, state);
			const compatResult = await runPatchPreflight(ctx);
			const patchSuffix = compatResult ? ` | patch: ${compatResult.status}` : "";
			const level = compatResult.status === "failed" || compatResult.status === "missing" || compatResult.status === "skipped" || !state.available ? "warning" : "info";
			ctx.ui.notify(`ccc: ${state.lastSummary}${patchSuffix}`, level);
		},
	});

	pi.registerCommand("ccc-patch", {
		description: "Apply the cocoindex compatibility patch and restart the ccc daemon",
		handler: async (_args, ctx) => {
			await runPatchPreflight(ctx, { force: true, manual: true });
			await refreshState(ctx, state);
		},
	});

	pi.registerTool({
		name: "search",
		label: "CocoIndex Search",
		description: "Semantic code search across the current codebase using cocoindex-code.",
		promptSnippet: "Use this to find code by meaning rather than exact string matches.",
		promptGuidelines: [
			"Use for conceptual code search, unfamiliar codebases, and locating implementations without exact identifiers.",
			"Start with a small limit, then paginate with offset if the first results are relevant.",
		],
		parameters: SEARCH_PARAMS,
		async execute(_toolCallId, rawParams, signal, onUpdate, ctx) {
			const params = rawParams as SearchParams;
			state.projectRoot = await findProjectRoot(process.cwd());
			let compatResult: PatchResult | null = await ensureCompatibilityPatch(state.projectRoot);
			if (compatResult.status === "patched") {
				onUpdate?.({ content: [{ type: "text", text: "Applied the ccc compatibility patch." }] });
			}
			state.available = true;
			state.busy = params.refresh_index === false ? "searching" : "indexing";
			renderStatus(ctx, state);

			const args = ["search"];
			if (params.refresh_index !== false) args.push("--refresh");
			for (const language of params.languages ?? []) args.push("--lang", language);
			for (const searchPath of params.paths ?? []) args.push("--path", searchPath);
			if (params.offset !== undefined) args.push("--offset", String(params.offset));
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			args.push(params.query);

			let commandResult: CommandResult;
			try {
				onUpdate?.({ content: [{ type: "text", text: `Running \`ccc ${args.join(" ")}\`` }] });
				commandResult = await runCommand("ccc", args, state.projectRoot, signal);
				if (commandResult.code !== 0 && isInitError(commandResult.combined)) {
					await ensureSearchReady(state.projectRoot, signal, (message) => {
						onUpdate?.({ content: [{ type: "text", text: message }] });
					});
					commandResult = await runCommand("ccc", args, state.projectRoot, signal);
				}

				if (commandResult.code !== 0) {
					throw new Error(buildCompatFailureMessage(commandResult.combined || "ccc search failed", compatResult));
				}

				const results = parseSearchResults(commandResult.stdout);
				state.initialized = true;
				state.lastSummary = `last search: ${results.length} result${results.length === 1 ? "" : "s"}`;
				state.busy = false;
				renderStatus(ctx, state);

				return {
					content: [{ type: "text", text: results.length > 0 ? `Found ${results.length} cocoindex result(s).` : "No results found." }],
					details: {
						success: true,
						results,
						total_returned: results.length,
						offset: params.offset ?? 0,
						message: null,
					},
				};
			} catch (error) {
				let message = error instanceof Error ? error.message : String(error);
				message = buildCompatFailureMessage(message, compatResult);
				state.busy = false;
				state.lastSummary = message;
				if ((error as NodeJS.ErrnoException)?.code === "ENOENT") {
					state.available = false;
					state.lastSummary = "ccc missing";
				}
				if (compatResult.status === "missing") {
					state.available = false;
				}
				state.initialized = false;
				renderStatus(ctx, state);

				return {
					content: [{ type: "text", text: `CocoIndex search failed: ${message}` }],
					details: {
						success: false,
						results: [],
						total_returned: 0,
						offset: params.offset ?? 0,
						message,
					},
				};
			}
		},
	});
}
