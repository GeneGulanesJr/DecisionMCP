/**
 * decisionmcp-lifecycle — start the DecisionMCP server with the first Pi session
 * and close it with the last one. Safe for multiple concurrent Pi instances.
 *
 * How it works
 * ------------
 * - `session_start`: ensure the server is healthy (spawn if not, under a
 *   lockfile so simultaneous Pis spawn exactly one), then register this
 *   session in `<repo>/data/decisionmcp-refs.json` as {token, pid}.
 * - `session_shutdown` + process `exit`: deregister; if no live refs remain,
 *   SIGTERM the server (pid from `data/decisionmcp.pid`, written only by us) and
 *   kick a final corpus backup.
 * - Entries whose pid is dead are pruned on every touch, so SIGKILLed Pi
 *   processes cannot leak refs (or keep the server up forever).
 *
 * A server started manually (`decisionmcp` in a terminal) has no pidfile from us
 * and is never killed by this extension.
 *
 * Install: symlink into ~/.pi/agent/extensions/decisionmcp-lifecycle.ts
 * (see integrations/pi/README.md).
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
	closeSync,
	openSync,
	readFileSync,
	rmSync,
	renameSync,
	statSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";

// --- Where things live (edit if the repo moves) ---------------------------
const REPO = "/Users/genejrgulanes/Documents/GulanesKorp/LayaMCP";
const BIN = `${REPO}/.venv/bin/decisionmcp`;
const PY = `${REPO}/.venv/bin/python`;
const DATA = `${REPO}/data`;
const REFS = `${DATA}/decisionmcp-refs.json`;
const LOCK = `${DATA}/decisionmcp-spawn.lock`;
const PIDF = `${DATA}/decisionmcp.pid`;
const LOG = `${DATA}/logs/decisionmcp.log`;
const HEALTH = "http://127.0.0.1:8765/health";
// --------------------------------------------------------------------------
const SPAWN_WAIT_MS = 120_000; // cold model preload can take a while
const LOCK_STALE_MS = 150_000; // > max possible spawn wait
const LOCK_POLL_MS = 50;

type Ref = { token: string; pid: number; at: number };

// Tiny sync sleep (blocks the loop; only ever a few ms).
const SAB = new Int32Array(new SharedArrayBuffer(4));
const sleepSync = (ms: number) => {
	try {
		Atomics.wait(SAB, 0, 0, ms);
	} catch {
		/* fall back to busy spin */
		const end = Date.now() + ms;
		while (Date.now() < end) {}
	}
};

const alive = (pid: number): boolean => {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
};

function writeRefs(refs: Ref[]): void {
	const tmp = `${REFS}.${process.pid}.tmp`;
	writeFileSync(tmp, JSON.stringify(refs));
	renameSync(tmp, REFS);
}

function readRefs(): Ref[] {
	try {
		const raw = JSON.parse(readFileSync(REFS, "utf8")) as Ref[];
		const live = Array.isArray(raw)
			? raw.filter((r) => r && Number.isFinite(r.pid) && alive(r.pid))
			: [];
		if (live.length !== raw.length) writeRefs(live); // prune dead pids
		return live;
	} catch {
		return [];
	}
}

/** Exclusive cross-process lock; steals it if the holder died mid-spawn. */
function acquireLock(timeoutMs = 10_000): boolean {
	const deadline = Date.now() + timeoutMs;
	for (;;) {
		try {
			const fd = openSync(LOCK, "wx");
			writeFileSync(fd, String(process.pid));
			closeSync(fd);
			return true;
		} catch (e: unknown) {
			const code = (e as { code?: string }).code;
			if (code !== "EEXIST") return false;
			try {
				if (Date.now() - statSync(LOCK).mtimeMs > LOCK_STALE_MS) {
					unlinkSync(LOCK); // stale: previous spawner died
					continue;
				}
			} catch {
				continue; // vanished; retry
			}
			if (Date.now() > deadline) return false;
			sleepSync(LOCK_POLL_MS);
		}
	}
}

function releaseLock(): void {
	try {
		unlinkSync(LOCK);
	} catch {
		/* not ours / already gone */
	}
}

async function healthy(ms = 1_500): Promise<boolean> {
	try {
		const r = await fetch(HEALTH, { signal: AbortSignal.timeout(ms) });
		return r.ok;
	} catch {
		return false;
	}
}

async function waitForHealth(ms: number): Promise<boolean> {
	const deadline = Date.now() + ms;
	while (Date.now() < deadline) {
		if (await healthy(1_000)) return true;
		await new Promise((r) => setTimeout(r, 500));
	}
	return false;
}

async function ensureRunning(): Promise<boolean> {
	if (await healthy()) return true;
	if (!acquireLock()) {
		// Another Pi is spawning right now; just wait for it.
		return waitForHealth(SPAWN_WAIT_MS);
	}
	try {
		if (await healthy()) return true; // double-check under the lock
		let fd: number;
		try {
			fd = openSync(LOG, "a");
		} catch {
			fd = "ignore" as unknown as number; // log dir missing; run quiet
		}
		const child = spawn(BIN, [], {
			cwd: REPO,
			detached: true, // outlive this Pi; refcount decides shutdown
			stdio: ["ignore", fd, fd],
		});
		child.unref();
		if (typeof child.pid === "number") writeFileSync(PIDF, String(child.pid));
		return waitForHealth(SPAWN_WAIT_MS);
	} finally {
		releaseLock();
	}
}

/** Deregister; if that was the last live session, stop the server + backup. */
function release(token: string): void {
	let refs: Ref[];
	if (acquireLock()) {
		try {
			refs = readRefs().filter((r) => r.token !== token);
			writeRefs(refs);
		} finally {
			releaseLock();
		}
	} else {
		// Lock held by a spawner; best-effort dereg without it.
		refs = readRefs().filter((r) => r.token !== token);
		writeRefs(refs);
	}
	if (refs.length > 0) return;
	try {
		const pid = Number.parseInt(readFileSync(PIDF, "utf8").trim(), 10);
		if (Number.isFinite(pid) && alive(pid)) process.kill(pid, "SIGTERM");
		rmSync(PIDF, { force: true });
	} catch {
		/* no pidfile: server wasn't spawned by this extension; leave it */
		return;
	}
	// Final capture of this session's corpus rows (server is shutting down).
	try {
		spawn(PY, ["scripts/backup_usage.py", "--keep", "14"], {
			cwd: REPO,
			detached: true,
			stdio: "ignore",
		}).unref();
	} catch {
		/* backup is best-effort */
	}
}

export default function decisionmcpLifecycle(pi: ExtensionAPI): void {
	const token = randomUUID();

	pi.on("session_start", async (_event, _ctx) => {
		try {
			await ensureRunning();
		} catch (e) {
			console.error("[decisionmcp-lifecycle] failed to ensure server:", e);
		}
		let ok = false;
		if (acquireLock()) {
			try {
				const refs = readRefs();
				if (!refs.some((r) => r.token === token))
					refs.push({ token, pid: process.pid, at: Date.now() });
				writeRefs(refs);
				ok = true;
			} finally {
				releaseLock();
			}
		}
		if (!ok) {
			// Could not take the lock in time; register best-effort anyway.
			const refs = readRefs();
			refs.push({ token, pid: process.pid, at: Date.now() });
			writeRefs(refs);
		}
	});

	pi.on("session_shutdown", () => {
		release(token);
	});

	// Cover abnormal exits (print mode, crashes): sync dereg on process exit.
	process.on("exit", () => {
		release(token);
	});
}
