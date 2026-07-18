import * as fs from "fs";
import * as path from "path";
import { spawnSync } from "child_process";
import { REPO_MANIFEST, type RepoGroupName } from "./repo-manifest.js";

interface Args {
  group: RepoGroupName | "all";
  update: boolean;
  depth: number;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  let group: RepoGroupName | "all" = "all";
  let update = false;
  let depth = 1;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--group" && args[i + 1]) {
      const v = args[i + 1];
      if (v !== "all" && v !== "type-defs" && v !== "usage") {
        throw new Error(`Invalid --group '${v}'. Use: all | type-defs | usage`);
      }
      group = v;
      i++;
    } else if (a === "--update") {
      update = true;
    } else if (a === "--depth" && args[i + 1]) {
      depth = Math.max(1, parseInt(args[i + 1], 10) || 1);
      i++;
    }
  }

  return { group, update, depth };
}

function runOrThrow(command: string, argv: string[], cwd?: string) {
  const res = spawnSync(command, argv, { stdio: "inherit", cwd });
  if (res.status !== 0) {
    throw new Error(`Command failed: ${command} ${argv.join(" ")}`);
  }
}

function main() {
  const { group, update, depth } = parseArgs();

  const targets = REPO_MANIFEST.filter((r) => group === "all" || r.group === group);
  console.log(`\nRepo fetch: group=${group}, count=${targets.length}, depth=${depth}`);

  for (const repo of targets) {
    const abs = path.resolve(repo.checkoutDir);
    const parent = path.dirname(abs);
    fs.mkdirSync(parent, { recursive: true });

    if (!fs.existsSync(abs)) {
      console.log(`\n[clone] ${repo.id}`);
      runOrThrow("git", ["clone", "--depth", String(depth), repo.url, abs]);
      continue;
    }

    if (update) {
      console.log(`\n[pull] ${repo.id}`);
      runOrThrow("git", ["pull", "--ff-only"], abs);
    } else {
      console.log(`\n[skip] ${repo.id} (already exists)`);
    }
  }

  console.log("\nDone fetching repositories.");
}

main();
