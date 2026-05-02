import { REPO_MANIFEST, type RepoGroupName } from "./repo-manifest.js";

function groupPaths(group: RepoGroupName): string[] {
  return REPO_MANIFEST.filter((r) => r.group === group).map((r) =>
    r.extractPath ? `${r.checkoutDir}/${r.extractPath}` : r.checkoutDir,
  );
}

export { type RepoGroupName };

export const REPO_GROUPS: Record<RepoGroupName, string[]> = {
  "type-defs": groupPaths("type-defs"),
  usage: groupPaths("usage"),
};
