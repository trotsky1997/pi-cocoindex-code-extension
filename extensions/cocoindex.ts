import { spawn } from "node:child_process";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const STATUS_KEY = "cocoindex";
const PATCH_SCRIPT_PATH = decodeURIComponent(new URL("../scripts/cocoindex_bm25_patch.py", import.meta.url).pathname);

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

type PatchStatus = "already_patched" | "failed" | "missing" | "patched" | "skipped";

type PatchResult = {
	status: PatchStatus;
	message: string;
	changedPaths: string[];
	scannedPaths: Record<string, string>;
};

type PatchScriptPayload = {
	status?: PatchStatus;
	message?: string;
	changed_paths?: string[];
	scanned_paths?: Record<string, string>;
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

function windowsExecutableExtensions(command: string): string[] {
	if (process.platform !== "win32" || path.extname(command)) {
		return [""];
	}
	const extensions = (process.env.PATHEXT || ".COM;.EXE;.BAT;.CMD")
		.split(";")
		.map((ext) => ext.trim())
		.filter(Boolean);
	return ["", ...extensions];
}

async function findExecutableCandidate(basePath: string): Promise<string | null> {
	for (const extension of windowsExecutableExtensions(basePath)) {
		const candidate = `${basePath}${extension}`;
		if (await pathExists(candidate)) {
			return candidate;
		}
	}
	return null;
}

async function findCommandPath(command: string): Promise<string | null> {
	const isPathLike = command.includes(path.sep) || command.includes("/") || command.includes("\\");
	if (isPathLike) {
		return findExecutableCandidate(command);
	}

	const pathValue = process.env.PATH ?? "";
	for (const dir of pathValue.split(path.delimiter)) {
		if (!dir) continue;
		const candidate = await findExecutableCandidate(path.join(dir, command));
		if (candidate) {
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

async function validateCccPython(pythonPath: string, projectRoot: string): Promise<string | null> {
	if (!(await pathExists(pythonPath))) {
		return null;
	}
	const check = await runCommand(pythonPath, ["-c", "import cocoindex_code, sys; print(sys.executable)"], projectRoot);
	return check.code === 0 ? pythonPath : null;
}

function pythonExecutablesForToolDir(toolDir: string): string[] {
	return [
		path.join(toolDir, "cocoindex-code", "Scripts", "python.exe"),
		path.join(toolDir, "cocoindex-code", "bin", "python3"),
		path.join(toolDir, "cocoindex-code", "bin", "python"),
	];
}

async function getUvToolDir(projectRoot: string): Promise<string | null> {
	const uvPath = await findCommandPath("uv");
	if (!uvPath) {
		return null;
	}
	const result = await runCommand(uvPath, ["tool", "dir"], projectRoot);
	if (result.code !== 0) {
		return null;
	}
	const lines = result.stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
	return lines[lines.length - 1] ?? null;
}

async function cccPythonCandidates(projectRoot: string, cccPath: string): Promise<string[]> {
	const candidates: string[] = [];
	const addToolDir = (toolDir: string | null | undefined) => {
		if (!toolDir) return;
		candidates.push(...pythonExecutablesForToolDir(toolDir));
	};

	addToolDir(await getUvToolDir(projectRoot));

	const cccDir = path.dirname(cccPath);
	addToolDir(path.resolve(cccDir, "..", "share", "uv", "tools"));
	addToolDir(path.resolve(cccDir, "..", "tools"));

	const home = process.env.USERPROFILE || process.env.HOME;
	if (home) {
		addToolDir(path.join(home, ".local", "share", "uv", "tools"));
		candidates.push(
			path.join(home, ".local", "pipx", "venvs", "cocoindex-code", "Scripts", "python.exe"),
			path.join(home, ".local", "pipx", "venvs", "cocoindex-code", "bin", "python"),
		);
	}

	for (const envRoot of [process.env.APPDATA, process.env.LOCALAPPDATA]) {
		if (!envRoot) continue;
		addToolDir(path.join(envRoot, "uv", "tools"));
		candidates.push(path.join(envRoot, "pipx", "venvs", "cocoindex-code", "Scripts", "python.exe"));
	}

	return [...new Set(candidates)];
}

async function resolveCccPython(projectRoot: string): Promise<string> {
	const cccPath = await findCommandPath("ccc");
	if (!cccPath) {
		throw new Error("ccc not found on PATH");
	}

	const launcherExt = path.extname(cccPath).toLowerCase();
	if (!new Set([".exe", ".cmd", ".bat"]).has(launcherExt)) {
		const launcherText = await readFile(cccPath, "utf8");
		const pythonHint = parseShebangInterpreter(launcherText);
		if (pythonHint) {
			const pythonPath = pythonHint.includes(path.sep) || pythonHint.includes("/") || pythonHint.includes("\\")
				? pythonHint
				: await findCommandPath(pythonHint);
			if (pythonPath) {
				const valid = await validateCccPython(pythonPath, projectRoot);
				if (valid) return valid;
			}
		}
	}

	for (const candidate of await cccPythonCandidates(projectRoot, cccPath)) {
		const valid = await validateCccPython(candidate, projectRoot);
		if (valid) return valid;
	}

	throw new Error(
		`Could not locate the Python environment behind ${cccPath}. ` +
			"Try running `uv tool dir` or reinstalling cocoindex-code with uv/pipx.",
	);
}

async function detectRapidFuzzVersion(pythonPath: string, projectRoot: string): Promise<string | null> {
	const check = await runCommand(pythonPath, ["-c", "import rapidfuzz; print(rapidfuzz.__version__)"], projectRoot);
	if (check.code !== 0) {
		return null;
	}
	const lines = check.stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
	return lines[lines.length - 1] ?? "installed";
}

async function ensureRapidFuzz(pythonPath: string, projectRoot: string): Promise<string> {
	const existing = await detectRapidFuzzVersion(pythonPath, projectRoot);
	if (existing) {
		return `RapidFuzz ${existing} is available for fuzzy reranking.`;
	}

	const uvPath = await findCommandPath("uv");
	if (!uvPath) {
		return "RapidFuzz is not installed; falling back to the built-in fuzzy reranker.";
	}

	const install = await runCommand(uvPath, ["pip", "install", "--python", pythonPath, "rapidfuzz"], projectRoot);
	if (install.code !== 0) {
		return `RapidFuzz install failed; using the built-in fuzzy reranker instead. ${install.combined || ""}`.trim();
	}

	const installed = await detectRapidFuzzVersion(pythonPath, projectRoot);
	if (installed) {
		return `Installed RapidFuzz ${installed} for BM25 reranking.`;
	}
	return "RapidFuzz install completed, but the module is still unavailable; using the built-in fuzzy reranker.";
}

function parsePatchPayload(output: string): PatchScriptPayload | null {
	for (const line of output.split(/\r?\n/).reverse()) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		try {
			return JSON.parse(trimmed) as PatchScriptPayload;
		} catch {
			continue;
		}
	}
	return null;
}

function normalizePatchResult(payload: PatchScriptPayload | null, fallbackMessage: string): PatchResult {
	return {
		status: payload?.status ?? "failed",
		message: payload?.message ?? fallbackMessage,
		changedPaths: payload?.changed_paths ?? [],
		scannedPaths: payload?.scanned_paths ?? {},
	};
}

async function applyBm25Patch(projectRoot: string): Promise<PatchResult> {
	const cccPath = await findCommandPath("ccc");
	if (!cccPath) {
		return {
			status: "missing",
			message: "ccc not found on PATH",
			changedPaths: [],
			scannedPaths: {},
		};
	}
	if (!(await pathExists(PATCH_SCRIPT_PATH))) {
		return {
			status: "failed",
			message: `Patch helper script not found at ${PATCH_SCRIPT_PATH}`,
			changedPaths: [],
			scannedPaths: {},
		};
	}

	try {
		const pythonPath = await resolveCccPython(projectRoot);
		const rapidFuzzMessage = await ensureRapidFuzz(pythonPath, projectRoot);
		const patchCommand = await runCommand(pythonPath, [PATCH_SCRIPT_PATH], projectRoot);
		const payload = parsePatchPayload(patchCommand.stdout);
		const result = normalizePatchResult(payload, patchCommand.combined || "BM25 patch failed");
		if (patchCommand.code !== 0 && result.status !== "patched") {
			return { ...result, status: "failed", message: `${result.message} ${rapidFuzzMessage}`.trim() };
		}
		const resultWithRapidFuzz = {
			...result,
			message: `${result.message} ${rapidFuzzMessage}`.trim(),
		};
		if (result.status === "patched") {
			const restart = await runCommand("ccc", ["daemon", "restart"], projectRoot);
			if (restart.code !== 0) {
				return {
					...resultWithRapidFuzz,
					message: `${resultWithRapidFuzz.message} Restarting the ccc daemon failed: ${restart.combined || "unknown error"}`,
				};
			}
		}
		return resultWithRapidFuzz;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		return {
			status: "failed",
			message: `BM25 patch failed: ${message}`,
			changedPaths: [],
			scannedPaths: {},
		};
	}
}

async function ensureCompatibilityPatch(projectRoot: string, options: { force?: boolean } = {}): Promise<PatchResult> {
	if (patchPromise) {
		return patchPromise;
	}
	if (!options.force && patchResult) {
		return patchResult;
	}

	patchPromise = applyBm25Patch(projectRoot)
		.then((result) => {
			patchResult = result;
			return result;
		})
		.finally(() => {
			patchPromise = null;
		});

	return patchPromise;
}

function appendPatchContext(message: string, compatResult: PatchResult | null): string {
	if (!compatResult) return message;
	if (compatResult.status === "patched" || compatResult.status === "already_patched") {
		return message;
	}
	return `${message}\nBM25 patch status: ${compatResult.message}`;
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

async function ensureSearchReady(projectRoot: string, signal: AbortSignal | undefined, onUpdate: (message: string) => void): Promise<void> {
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
			ctx.ui.notify(`ccc bm25: ${result.message}`, "info");
			return;
		}
		if (manual) {
			const level = result.status === "failed" || result.status === "missing" || result.status === "skipped" ? "warning" : "info";
			ctx.ui.notify(`ccc bm25: ${result.message}`, level);
			return;
		}
		if (result.status === "failed" || result.status === "skipped") {
			ctx.ui.notify(`ccc bm25: ${result.message}`, "warning");
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
			ctx.ui.notify(`ccc bm25: ${message}`, "warning");
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
		description: "Refresh the cocoindex status indicator and BM25 patch status",
		handler: async (_args, ctx) => {
			state.busy = "checking";
			renderStatus(ctx, state);
			await refreshState(ctx, state);
			state.busy = false;
			renderStatus(ctx, state);
			const compatResult = await runPatchPreflight(ctx);
			const level = compatResult.status === "failed" || compatResult.status === "missing" || compatResult.status === "skipped" || !state.available ? "warning" : "info";
			ctx.ui.notify(`ccc: ${state.lastSummary} | bm25 patch: ${compatResult.status}`, level);
		},
	});

	pi.registerCommand("ccc-patch", {
		description: "Patch cocoindex-code into BM25 mode and restart the ccc daemon",
		handler: async (_args, ctx) => {
			await runPatchPreflight(ctx, { force: true, manual: true });
			await refreshState(ctx, state);
		},
	});

	pi.registerTool({
		name: "search",
		label: "CocoIndex Search",
		description: "Code search across the current codebase using cocoindex-code.",
		promptSnippet: "Use this to search the codebase through cocoindex-code's BM25-aware search backend.",
		promptGuidelines: [
			"Use for code search, implementation lookup, and retrieval after the extension patches cocoindex-code into BM25 mode.",
			"Start with a small limit, then paginate with offset if the first results are relevant.",
		],
		parameters: SEARCH_PARAMS,
		async execute(_toolCallId, rawParams, signal, onUpdate, ctx) {
			const params = rawParams as SearchParams;
			state.projectRoot = await findProjectRoot(process.cwd());
			let compatResult: PatchResult | null = await ensureCompatibilityPatch(state.projectRoot);
			if (compatResult.status === "patched") {
				onUpdate?.({ content: [{ type: "text", text: "Enabled BM25 mode for cocoindex-code." }] });
			}
			state.available = compatResult.status !== "missing";
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
					throw new Error(appendPatchContext(commandResult.combined || "ccc search failed", compatResult));
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
				const message = appendPatchContext(error instanceof Error ? error.message : String(error), compatResult);
				state.busy = false;
				state.lastSummary = message;
				if ((error as NodeJS.ErrnoException)?.code === "ENOENT") {
					state.available = false;
					state.lastSummary = "ccc missing";
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
