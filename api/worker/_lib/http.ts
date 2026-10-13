import { BrokerError, env } from "./protocol.js";

export async function fetchWithRetry(input: string, init: RequestInit = {}): Promise<Response> {
  // Retrying a POST can repeat a device-code, assignment, or mutation after the
  // server has already applied it.  Only retry methods whose requests are safe
  // to replay; callers that need mutation retries must provide their own CAS.
  const requestMethod = (init.method || "GET").toUpperCase();
  const safe = requestMethod === "GET" || requestMethod === "HEAD" || requestMethod === "OPTIONS";
  if (!safe) return fetch(input, init);
  let response: Response | undefined;
  for (let attempt = 0; attempt < 3; attempt++) {
    init.signal?.throwIfAborted();
    try { response = await fetch(input, init); } catch (error) {
      if (attempt === 2 || init.signal?.aborted) throw error;
      await retryDelay(100 * (attempt + 1), init.signal); continue;
    }
    if (![408, 425, 429, 500, 502, 503, 504].includes(response.status) || attempt === 2) return response;
    const retryAfter = response.headers.get("retry-after");
    const seconds = retryAfter && /^\d+$/.test(retryAfter) ? Math.min(30, Number(retryAfter)) : 0;
    await retryDelay(Math.max(100 * (attempt + 1), seconds * 1000), init.signal);
  }
  return response as Response;
}

async function retryDelay(ms: number, signal?: AbortSignal | null): Promise<void> {
  if (!signal) { await new Promise((resolve) => setTimeout(resolve, ms)); return; }
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const onAbort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", onAbort); resolve(); }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
}
