// Reading a Server-Sent Events stream from a POST endpoint. EventSource is GET-only, so
// we POST with fetch and parse the `event:` / `data:` frames off the response body
// ourselves. Shared by the train and synthetic-generation stores.

function parseFrame(frame: string): { event: string; data: string } {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return { event, data };
}

/** POST `body` to `url` and invoke `onEvent(event, data)` for each SSE frame until the
 * stream ends. Throws on a non-OK response (carrying the backend's `detail` when present)
 * or a transport error; an aborted `signal` surfaces as an AbortError for the caller to
 * ignore. */
export async function readSSE(
  url: string,
  body: unknown,
  signal: AbortSignal,
  onEvent: (event: string, data: Record<string, unknown>) => void,
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err?.detail) detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch {
      // non-JSON error body — keep the status line
    }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const { event, data } = parseFrame(frame);
      if (!data) continue;
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(data) as Record<string, unknown>;
      } catch {
        continue;
      }
      onEvent(event, payload);
    }
  }
}
