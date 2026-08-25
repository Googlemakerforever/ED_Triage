"use client";

import { useApiGet } from "@/hooks/use-api-get";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonCard } from "@/components/states/loaders";

type ErrorLog = { id: string; section: string; topic: string; mistakeType: string; status: string; recurrenceCount: number }[];

export default function ErrorLogPage() {
  const { data, loading, error } = useApiGet<ErrorLog>("/api/error-log");

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;
  if (!data?.length) return <EmptyState title="No error log entries" body="Add mistakes from review to start pattern tracking." />;

  return (
    <SectionPage title="Error Log" subtitle="Track recurring mistakes by section, topic, and cause.">
      <div className="space-y-3">
        {data.map((entry) => <Card key={entry.id}><p className="font-semibold">{entry.topic}</p><p className="text-sm text-slate-600">{entry.section} • {entry.mistakeType} • {entry.status} • repeats {entry.recurrenceCount}</p></Card>)}
      </div>
    </SectionPage>
  );
}
