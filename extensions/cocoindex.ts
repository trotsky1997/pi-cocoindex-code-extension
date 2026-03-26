import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";

import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const STATUS_KEY = "cocoindex";

type StatusState = {
	available: boolean;
	busy: false | "checking" | "indexing" | "searching";
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

	pi.on("session_start", async (_event, ctx) => {
		state.busy = "checking";
		renderStatus(ctx, state);
		await refreshState(ctx, state);
		state.busy = false;
		renderStatus(ctx, state);
	});

	pi.on("session_switch", async (event, ctx) => {
		if (event.reason !== "new") return;
		state.busy = "checking";
		renderStatus(ctx, state);
		await refreshState(ctx, state);
		state.busy = false;
		renderStatus(ctx, state);
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
			ctx.ui.notify(`ccc: ${state.lastSummary}`, state.available ? "info" : "warning");
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
					throw new Error(commandResult.combined || "ccc search failed");
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
				state.busy = false;
				state.lastSummary = error instanceof Error ? error.message : "search failed";
				if ((error as NodeJS.ErrnoException)?.code === "ENOENT") {
					state.available = false;
					state.lastSummary = "ccc missing";
				}
				state.initialized = false;
				renderStatus(ctx, state);

				const message = error instanceof Error ? error.message : String(error);
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
