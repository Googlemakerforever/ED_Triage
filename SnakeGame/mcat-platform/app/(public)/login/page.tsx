"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { apiPost } from "@/lib/api/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      await apiPost("/api/auth/login", { email, password });
      router.push("/app/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to login");
    }
  }

  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <Card>
        <h1 className="text-2xl font-bold">Log in</h1>
        <form className="mt-4 space-y-3" onSubmit={onSubmit}>
          <label className="text-sm">Email<Input required type="email" value={email} onChange={(e) => setEmail(e.target.value)} /></label>
          <label className="text-sm">Password<Input required type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
          {error && <p className="text-sm text-rose-600">{error}</p>}
          <Button type="submit" className="w-full">Log in</Button>
        </form>
        <Link href="/forgot-password" className="mt-3 block text-sm text-brand-700">Forgot password?</Link>
      </Card>
    </div>
  );
}
