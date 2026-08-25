export async function apiPost<TBody, TResp>(url: string, body: TBody): Promise<TResp> {
  const response = await fetch(url, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload?.error?.message ?? "Request failed");
  }

  return payload.data as TResp;
}

export async function apiPatch<TBody, TResp>(url: string, body: TBody): Promise<TResp> {
  const response = await fetch(url, {
    method: "PATCH",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload?.error?.message ?? "Request failed");
  }

  return payload.data as TResp;
}
