"use client";

import Link from "next/link";
import { useApiGet } from "@/hooks/use-api-get";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonCard } from "@/components/states/loaders";

type Topic = { id: string; slug: string; title: string; section: string; summary: string; highYield: boolean }[];

export default function ContentPage() {
  const { data, loading, error } = useApiGet<Topic>("/api/content/topics");

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;
  if (!data?.length) return <EmptyState title="No topics yet" body="Seed or import content to enable library browsing." />;

  return (
    <SectionPage title="Content Library" subtitle="High-yield topic map with mastery tracking and linked practice.">
      <div className="grid gap-3 md:grid-cols-2">
        {data.map((topic) => (
          <Link key={topic.id} href={`/app/content/${topic.slug}`}>
            <Card>
              <p className="font-semibold">{topic.title}</p>
              <p className="text-sm text-slate-600">{topic.section} {topic.highYield ? "• High yield" : ""}</p>
              <p className="mt-2 text-sm text-slate-700">{topic.summary}</p>
            </Card>
          </Link>
        ))}
      </div>
    </SectionPage>
  );
}
