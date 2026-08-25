import { ReactNode } from "react";

export function SectionPage({ title, subtitle, actions, children }: { title: string; subtitle: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">{title}</h1>
          <p className="text-sm text-slate-600">{subtitle}</p>
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}
