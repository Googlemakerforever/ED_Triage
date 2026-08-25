"use client";

import { useApiGet } from "@/hooks/use-api-get";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonCard } from "@/components/states/loaders";

type Question = { id: string; stem: string; difficulty: number; topic: { title: string; section: string } };

export function PracticeHubView() {
  const { data, loading, error } = useApiGet<Question[]>("/api/practice/questions");

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;
  if (!data?.length) return <EmptyState title="No questions available" body="Try adjusting filters or seeding data." />;

  return (
    <SectionPage title="Practice Hub" subtitle="Question and passage modes with timed options and filters.">
      <div className="grid gap-3">
        {data.slice(0, 8).map((q) => (
          <Card key={q.id}>
            <p className="font-semibold">{q.stem}</p>
            <p className="mt-1 text-sm text-slate-600">{q.topic.section} • {q.topic.title} • Difficulty {q.difficulty}</p>
          </Card>
        ))}
      </div>
    </SectionPage>
  );
}
