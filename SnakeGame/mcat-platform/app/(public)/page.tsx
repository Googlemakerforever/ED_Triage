import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export default function LandingPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16">
      <section className="grid gap-10 md:grid-cols-2 md:items-center">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-wide text-brand-700">Built for 520+ ambition</p>
          <h1 className="text-4xl font-extrabold leading-tight text-slate-900 md:text-5xl">MCAT prep that actually diagnoses, adapts, and improves decision quality.</h1>
          <p className="mt-5 text-lg text-slate-600">AegisMCAT combines real diagnostic scoring, adaptive planning, deep review, analytics, and personalized AI tutoring.</p>
          <div className="mt-8 flex gap-3">
            <Link href="/signup"><Button>Start free trial</Button></Link>
            <Link href="/features"><Button variant="secondary">Explore platform</Button></Link>
          </div>
        </div>
        <Card className="space-y-4">
          <h2 className="text-lg font-semibold">Live Product Systems</h2>
          <ul className="space-y-2 text-sm text-slate-700">
            <li>Diagnostic engine with content/reasoning/execution miss typing</li>
            <li>Practice sessions with timing + confidence capture</li>
            <li>Review engine with transfer rules and error logging</li>
            <li>Analytics for score projection, timing, and classification trends</li>
            <li>AI tutor with teach, hint, review, Socratic, drill, CARS, and planning modes</li>
          </ul>
        </Card>
      </section>
    </div>
  );
}
