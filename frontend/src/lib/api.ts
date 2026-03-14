const BASE = '';

export async function api<T = any>(
  url: string,
  opts: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers as any },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

export async function apiRaw(url: string, opts: RequestInit = {}): Promise<Response> {
  return fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers as any },
    ...opts,
  });
}

// Upload files (FormData, no JSON header)
export async function apiUpload<T = any>(
  url: string,
  formData: FormData
): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}
