"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const nav = [
  ["Dashboard", "/app/dashboard"],
  ["Diagnostic", "/app/diagnostic"],
  ["Study Plan", "/app/study-plan"],
  ["Practice Hub", "/app/practice"],
  ["Passage Practice", "/app/passages"],
  ["Review", "/app/review"],
  ["Error Log", "/app/error-log"],
  ["Spaced Repetition", "/app/spaced-repetition"],
  ["Analytics", "/app/analytics"],
  ["AI Tutor", "/app/ai-tutor"],
  ["Content Library", "/app/content"],
  ["Settings", "/app/settings"]
] as const;

export function AppShell({ children, userName }: { children: ReactNode; userName: string }) {
  const pathname = usePathname();
  async function logout() {
    await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
    window.location.href = "/login";
  }

  return (
    <div className="min-h-screen md:grid md:grid-cols-[260px_1fr]">
      <aside className="hidden border-r border-slate-200 bg-white p-4 md:block">
        <div className="mb-6 text-lg font-bold">AegisMCAT</div>
        <nav className="space-y-1">
          {nav.map(([label, href]) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "block rounded-lg px-3 py-2 text-sm",
                pathname === href ? "bg-brand-100 text-brand-900" : "text-slate-700 hover:bg-slate-100"
              )}
            >
              {label}
            </Link>
          ))}
        </nav>
      </aside>
      <main>
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="text-sm text-slate-600">Notifications and reminders coming soon</div>
            <div className="hidden text-sm text-slate-500 md:block">|</div>
            <div className="hidden text-sm font-medium text-slate-900 md:block">{userName}</div>
          </div>
          <Button variant="secondary" onClick={logout}>Logout</Button>
        </header>
        <nav className="flex gap-2 overflow-x-auto border-b border-slate-200 bg-white px-4 py-2 md:hidden">
          {nav.map(([label, href]) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "whitespace-nowrap rounded-md px-2 py-1 text-xs",
                pathname === href ? "bg-brand-100 text-brand-900" : "bg-slate-100 text-slate-700"
              )}
            >
              {label}
            </Link>
          ))}
        </nav>
        <div className="mx-auto max-w-7xl p-4 md:p-6">{children}</div>
      </main>
    </div>
  );
}
