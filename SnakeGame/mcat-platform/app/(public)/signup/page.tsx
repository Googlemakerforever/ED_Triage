"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiPost } from "@/lib/api/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function SignupPage() {
  const router = useRouter();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      await apiPost("/api/auth/signup", { fullName, email, password });
      router.push("/app/onboarding");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create account");
    }
  }

  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <Card>
        <h1 className="text-2xl font-bold">Create account</h1>
        <form className="mt-4 space-y-3" onSubmit={onSubmit}>
          <label className="text-sm">Full name<Input required value={fullName} onChange={(e) => setFullName(e.target.value)} /></label>
          <label className="text-sm">Email<Input required type="email" value={email} onChange={(e) => setEmail(e.target.value)} /></label>
          <label className="text-sm">Password<Input required type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
          {error && <p className="text-sm text-rose-600">{error}</p>}
          <Button type="submit" className="w-full">Start setup</Button>
        </form>
      </Card>
    </div>
  );
}
