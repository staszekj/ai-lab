/**
 * Shared siblings + containing-decl helpers.
 *
 * Used by:
 *   - extract.ts          (training-time annotation record)
 *   - refiner-locate.ts   (inference-time candidate record)
 *
 * The two pipelines MUST produce identical siblings strings for the same
 * code position, otherwise the model sees a different distribution at
 * inference than during training (one of the largest known sources of
 * accuracy regression).
 *
 * Output shape:
 *   ""                                       → no useful context
 *   "[name: T, name: T, -> R]"               → params + return type
 *   "[@in:Foo, name: T]"                     → containing-decl boost prefix
 *
 * Limits: ≤12 entries, ≤250 characters (incl. brackets). Entries past the
 * limit are dropped silently.
 */

import { Node, SyntaxKind } from "ts-morph";

const MAX_ENTRIES = 12;
const MAX_CHARS = 250;

// Rules whose target identifier is typically declared elsewhere in the file
// (so the model benefits from knowing the enclosing decl's name). Plan §1.1.
function shouldBoost(rule: string): boolean {
  const key = rule.split("→")[0];
  if (key === "indexed_access_type") return true;
  if (key === "utility_type") return true;
  if (key === "conditional_type") return true;
  if (key.includes("setstateaction")) return true;
  if (/^react_.*_props(_|$)/.test(key)) return true;
  return false;
}

function packEntries(entries: readonly string[]): string {
  if (entries.length === 0) return "";
  const limited = entries.slice(0, MAX_ENTRIES);
  const body = limited.join(", ");
  const out = "[" + body + "]";
  if (out.length <= MAX_CHARS) return out;
  // Drop entries from the tail until we fit.
  let kept = limited.slice();
  while (kept.length > 0) {
    kept.pop();
    const candidate = "[" + kept.join(", ") + "]";
    if (candidate.length <= MAX_CHARS) return candidate;
  }
  return "";
}

function fnReturnTypeText(fn: Node): string | null {
  if (
    Node.isFunctionDeclaration(fn) ||
    Node.isArrowFunction(fn) ||
    Node.isFunctionExpression(fn) ||
    Node.isMethodDeclaration(fn) ||
    Node.isMethodSignature(fn) ||
    Node.isFunctionTypeNode(fn)
  ) {
    const rt = fn.getReturnTypeNode();
    return rt ? rt.getText() : null;
  }
  return null;
}

function paramEntries(fn: Node, exclude?: Node): string[] {
  if (
    !(
      Node.isFunctionDeclaration(fn) ||
      Node.isArrowFunction(fn) ||
      Node.isFunctionExpression(fn) ||
      Node.isMethodDeclaration(fn) ||
      Node.isConstructorDeclaration(fn) ||
      Node.isMethodSignature(fn) ||
      Node.isFunctionTypeNode(fn)
    )
  ) {
    return [];
  }
  const out: string[] = [];
  for (const p of fn.getParameters()) {
    if (p === exclude) continue;
    const tn = p.getTypeNode();
    const t = tn ? tn.getText() : "?";
    out.push(`${p.getName()}: ${t}`);
  }
  return out;
}

function peerPropertyEntries(prop: Node): string[] {
  const parent = prop.getParent();
  if (
    !parent ||
    !(
      Node.isInterfaceDeclaration(parent) ||
      Node.isTypeLiteral(parent) ||
      Node.isClassDeclaration(parent)
    )
  ) {
    return [];
  }
  const out: string[] = [];
  for (const m of parent.getMembers()) {
    if (m === prop) continue;
    if (Node.isPropertySignature(m) || Node.isPropertyDeclaration(m)) {
      const tn = m.getTypeNode();
      const t = tn ? tn.getText() : "?";
      out.push(`${m.getName()}: ${t}`);
    }
  }
  return out;
}

function destructurePatternEntries(varDecl: Node): string[] {
  if (!Node.isVariableDeclaration(varDecl)) return [];
  const nameNode = varDecl.getNameNode();
  if (
    !Node.isObjectBindingPattern(nameNode) &&
    !Node.isArrayBindingPattern(nameNode)
  ) {
    return [];
  }
  const out: string[] = [];
  for (const el of nameNode.getElements()) {
    if (!Node.isBindingElement(el)) continue;
    const name = el.getName();
    // The declared type lives on the VariableDeclaration as a whole, so we
    // can only emit names here. Still useful: model sees the destructured
    // property set even if the value type is degraded to `unknown`.
    out.push(name);
  }
  return out;
}

function genericArgEntries(call: Node, exclude: Node): string[] {
  if (!Node.isCallExpression(call)) return [];
  const out: string[] = [];
  for (const ta of call.getTypeArguments()) {
    if (ta === exclude) continue;
    out.push(ta.getText());
  }
  return out;
}

/**
 * Find the closest enclosing **named** declaration whose name is useful as a
 * locator hint for cross-decl rules. Walks up parents looking for:
 *   - InterfaceDeclaration / TypeAliasDeclaration / ClassDeclaration
 *   - Named FunctionDeclaration / MethodDeclaration
 *   - VariableDeclaration when its initializer is a function/arrow
 *     (i.e., a React-style component or named lambda).
 */
export function getContainingDeclName(node: Node): string | null {
  let cur: Node | undefined = node.getParent();
  while (cur) {
    if (Node.isInterfaceDeclaration(cur)) return cur.getName();
    if (Node.isTypeAliasDeclaration(cur)) return cur.getName();
    if (Node.isClassDeclaration(cur)) return cur.getName() ?? null;
    if (Node.isFunctionDeclaration(cur)) {
      const n = cur.getName();
      if (n) return n;
    }
    if (Node.isMethodDeclaration(cur)) {
      const n = cur.getName();
      if (n) return n;
    }
    if (
      (Node.isArrowFunction(cur) || Node.isFunctionExpression(cur)) &&
      Node.isVariableDeclaration(cur.getParent() ?? undefined as unknown as Node)
    ) {
      const vd = cur.getParent() as import("ts-morph").VariableDeclaration;
      const n = vd.getName();
      if (n) return n;
    }
    cur = cur.getParent();
  }
  return null;
}

/**
 * Build the enriched siblings string for the declaration `decl` (the node
 * that owns the type annotation we're refining), branching on `kind`.
 *
 * NOTE: containing-decl boost is NOT applied here — it depends on the rule,
 * which is known only at degrade / locate time. Call `applyContainingBoost`
 * to layer it on top.
 */
export function buildSiblings(decl: Node, kind: string): string {
  let entries: string[] = [];

  switch (kind) {
    case "parameter": {
      const param = decl as import("ts-morph").ParameterDeclaration;
      const fn = param.getParent();
      if (fn) {
        entries = paramEntries(fn, param);
        const rt = fnReturnTypeText(fn);
        if (rt) entries.push(`-> ${rt}`);
      }
      break;
    }
    case "property": {
      entries = peerPropertyEntries(decl);
      break;
    }
    case "variable": {
      entries = destructurePatternEntries(decl);
      break;
    }
    case "return_type": {
      entries = paramEntries(decl);
      break;
    }
    case "generic_argument": {
      // `decl` is the type argument; `decl.getParent()` chain finds the call.
      let call: Node | undefined = decl.getParent();
      while (call && !Node.isCallExpression(call)) call = call.getParent();
      if (call) entries = genericArgEntries(call, decl);
      break;
    }
    case "type_assertion": {
      if (Node.isAsExpression(decl)) {
        const expr = decl.getExpression().getText();
        if (expr.length <= 80) entries = [`expr:${expr}`];
      }
      break;
    }
    default:
      entries = [];
  }

  return packEntries(entries);
}

/**
 * Conditionally prepend `@in:<containingDecl>` inside the brackets when the
 * rule benefits from a cross-decl locator hint. If `siblings` is empty,
 * still emit `[@in:Name]` so the model gets the boost.
 */
export function applyContainingBoost(
  siblings: string,
  containingDecl: string | null,
  rule: string,
): string {
  if (!containingDecl) return siblings;
  if (!shouldBoost(rule)) return siblings;

  const boost = `@in:${containingDecl}`;
  if (!siblings) return `[${boost}]`;
  // siblings is "[...]" — splice the boost in as the first entry.
  if (siblings.startsWith("[") && siblings.endsWith("]")) {
    const inner = siblings.slice(1, -1);
    const merged = inner.length === 0 ? boost : `${boost}, ${inner}`;
    const out = `[${merged}]`;
    if (out.length <= MAX_CHARS) return out;
    return `[${boost}]`;
  }
  return siblings;
}
