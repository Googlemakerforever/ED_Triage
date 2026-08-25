"use client";

import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonCard } from "@/components/states/loaders";
import { useApiGet } from "@/hooks/use-api-get";

type Queue = { dueToday: { id: string; prompt: string; topic: string }[]; overdueCount: number };

export default function SpacedRepetitionPage() {
  const { data, loading, error } = useApiGet<Queue>("/api/spaced-repetition/queue");

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;
  if (!data?.dueToday.length) return <EmptyState title="No cards due" body="Great momentum. Review queue will repopulate automatically." />;

  return (
    <SectionPage title="Spaced Repetition" subtitle="Daily review queue to strengthen recall and prevent repeat misses.">
      <Card><p className="text-sm text-slate-600">Overdue: {data.overdueCount}</p></Card>
      <div className="space-y-3">{data.dueToday.map((card) => <Card key={card.id}><p className="font-semibold">{card.prompt}</p><p className="text-sm text-slate-600">{card.topic}</p></Card>)}</div>
    </SectionPage>
  );
}
