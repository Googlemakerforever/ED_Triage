import Link from "next/link";
import { Button } from "@/components/ui/button";

export function PublicNav() {
  return (
    <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
        <Link href="/" className="text-lg font-bold text-slate-900">AegisMCAT</Link>
        <nav className="hidden gap-6 text-sm text-slate-700 md:flex">
          <Link href="/features">Features</Link>
          <Link href="/methodology">Methodology</Link>
          <Link href="/pricing">Pricing</Link>
        </nav>
        <div className="flex items-center gap-2">
          <Link href="/login"><Button variant="ghost">Log in</Button></Link>
          <Link href="/signup"><Button>Start free</Button></Link>
        </div>
      </div>
    </header>
  );
}
