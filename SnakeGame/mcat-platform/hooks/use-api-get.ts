"use client";

import { useEffect, useState } from "react";

type State<T> = { data: T | null; loading: boolean; error: string | null };

export function useApiGet<T>(url: string) {
  const [state, setState] = useState<State<T>>({ data: null, loading: true, error: null });

  useEffect(() => {
    if (!url || url === "__disabled__") {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let active = true;
    async function run() {
      setState({ data: null, loading: true, error: null });
      try {
        const response = await fetch(url, { credentials: "include" });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload?.error?.message ?? "Request failed");
        }
        if (active) setState({ data: payload.data as T, loading: false, error: null });
      } catch (error) {
        if (active) setState({ data: null, loading: false, error: error instanceof Error ? error.message : "Unexpected error" });
      }
    }
    void run();

    return () => {
      active = false;
    };
  }, [url]);

  return state;
}
