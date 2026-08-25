"use client";

import { useParams } from "next/navigation";
import { useApiGet } from "@/hooks/use-api-get";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { ErrorState, SkeletonCard } from "@/components/states/loaders";

export default function TopicDetailPage() {
  const params = useParams<{ slug: string }>();
  const slug = params?.slug;
  const { data, loading, error } = useApiGet<{ title: string; section: string; summary: string; contentMd: string }>(
    slug ? `/api/content/topics/${slug}` : "/api/content/topics/unknown"
  );

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;
  if (!data) return <ErrorState message="Topic unavailable" />;

  return (
    <SectionPage title={data.title} subtitle={`${data.section} topic`}>
      <Card>
        <p className="text-sm text-slate-700">{data.summary}</p>
        <p className="mt-4 whitespace-pre-wrap text-sm text-slate-700">{data.contentMd}</p>
      </Card>
    </SectionPage>
  );
}
