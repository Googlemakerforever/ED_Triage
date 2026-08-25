import { Card } from "@/components/ui/card";

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16 space-y-4">
      <h1 className="text-3xl font-bold">Methodology</h1>
      <p className="text-slate-600">We model MCAT performance as an interaction of knowledge, reasoning quality, and execution under time pressure.</p>
      <div className="grid gap-4 md:grid-cols-3">
        <Card><h2 className="font-semibold">1. Diagnose Precisely</h2><p className="mt-2 text-sm text-slate-700">Every session captures right/wrong, confidence, timing, and miss class to expose true limiting factors.</p></Card>
        <Card><h2 className="font-semibold">2. Train Adaptively</h2><p className="mt-2 text-sm text-slate-700">Planner and practice assignment shift based on weak skills, retention risk, and exam timeline.</p></Card>
        <Card><h2 className="font-semibold">3. Review for Transfer</h2><p className="mt-2 text-sm text-slate-700">Explanations focus on logic transfer rules so mistakes become repeatable wins on new items.</p></Card>
      </div>
    </div>
  );
}
