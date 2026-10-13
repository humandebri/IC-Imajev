/** Opaque canister-produced carries: no model arithmetic in JavaScript. */
export interface Header {
  version: number; model: string; pack_hash: string; input_hash: string;
  step: number; op: string; encoding: string; tensor: string;
  aux: string[]; dims: number[]; scalars: number[];
}
export const C = 2560, H = 9216, P = 5, CONV = 24576;
export function concat(...parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.length; }
  return out;
}
export function u32(n: number) {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, n, true);
  return bytes;
}
export async function sha(bytes: Uint8Array): Promise<string> {
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.slice().buffer));
  return Array.from(hash, b => b.toString(16).padStart(2, "0")).join("");
}
function headerText(h: Header) {
  // Match the reference Python's F32 scalar spelling for byte parity.
  return JSON.stringify(h).replace('"scalars":[2,0.000001]', '"scalars":[2.0,1e-06]');
}
export async function frame(h: Header, payload: Uint8Array) {
  const header = new TextEncoder().encode(headerText(h));
  const body = concat(u32(header.length), header, payload);
  if (header.length > 16384 || body.length + 32 > 1_990_000) throw new Error("Query frame too large.");
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", body.buffer));
  return concat(body, digest);
}
export async function unframe(bytes: Uint8Array, expected: Header, verifiedAgentReply = false) {
  if (bytes.length < 36 || bytes.length > 1_990_000) throw new Error("Invalid query reply size.");
  const size = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  if (size > 16384 || size + 36 > bytes.length) throw new Error("Invalid reply header.");
  const body = bytes.subarray(0, -32);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", body.slice().buffer));
  const footer = bytes.subarray(-32);
  // Host-checksum runtime leaves a zero footer. Only accept that from a
  // query client that verified the IC node signature, then bind every field.
  const hostBound = verifiedAgentReply && footer.every(b => b === 0);
  if (!hostBound && !digest.every((b, i) => b === footer[i])) throw new Error("Query reply checksum mismatch.");
  const h = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + size))) as Header;
  for (const key of Object.keys(expected) as (keyof Header)[]) {
    const actual = key === "scalars" ? h.scalars?.map(Math.fround) : h[key];
    const wanted = key === "scalars" ? expected.scalars.map(Math.fround) : expected[key];
    if (JSON.stringify(actual) !== JSON.stringify(wanted)) throw new Error(`Query reply ${key} mismatch.`);
  }
  if (hostBound) bytes.set(digest, bytes.length - 32);
  return bytes.subarray(4 + size, -32);
}
export function front(layer: number) {
  return layer < 3 ? 5120 - layer * 256 : [5120, 4864, 4608, 4352][(layer - 3) % 4];
}
export function checkCarry(payload: Uint8Array, n: number, done: number) {
  const length = 5 + n * (3 * C + done + 4 * (C / 256 + done / 256 + 192));
  if (payload.length !== length || payload[0] !== 1 ||
      new DataView(payload.buffer, payload.byteOffset + 1, 4).getUint32(0, true) !== done) {
    throw new Error("Invalid MLP carry shape or progress.");
  }
}
