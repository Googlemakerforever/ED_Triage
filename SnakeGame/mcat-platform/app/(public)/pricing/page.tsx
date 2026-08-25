import { Card } from "@/components/ui/card";

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16">
      <h1 className="text-3xl font-bold">Pricing</h1>
      <p className="mt-2 text-slate-600">Built for serious prep windows from baseline recovery to 523+ optimization.</p>
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <Card><h3 className="font-semibold">Focused</h3><p className="mt-2 text-sm text-slate-600">Core diagnostics, planner, and review systems.</p><p className="mt-4 text-2xl font-bold">$39/mo</p></Card>
        <Card className="border-brand-500"><h3 className="font-semibold">Performance</h3><p className="mt-2 text-sm text-slate-600">Full analytics, AI tutor modes, and adaptive practice.</p><p className="mt-4 text-2xl font-bold">$89/mo</p></Card>
        <Card><h3 className="font-semibold">Elite</h3><p className="mt-2 text-sm text-slate-600">High-volume workflow and advanced planning support.</p><p className="mt-4 text-2xl font-bold">$149/mo</p></Card>
      </div>
    </div>
  );
}
