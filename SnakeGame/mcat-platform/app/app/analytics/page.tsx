"use client";

import { useApiGet } from "@/hooks/use-api-get";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonCard } from "@/components/states/loaders";

type Trend = { date: string; accuracy: number; avgTime: number }[];
type Overview = {
  prediction: { low: number; high: number } | null;
  confidenceVsAccuracy: { avgConfidence: number; accuracy: number };
  timingVsAccuracy: { avgTimeSeconds: number; accuracy: number };
  classifications: { contentGaps: number; reasoningGaps: number; executionGaps: number };
};

export default function AnalyticsPage() {
  const trends = useApiGet<Trend>("/api/analytics/trends?days=30");
  const overview = useApiGet<Overview>("/api/analytics/overview");

  if (trends.loading || overview.loading) return <SkeletonCard />;
  if (trends.error) return <ErrorState message={trends.error} />;
  if (overview.error) return <ErrorState message={overview.error} />;
  if (!trends.data?.length || !overview.data) return <EmptyState title="No analytics yet" body="Complete sessions to generate trend lines." />;

  return (
    <SectionPage title="Analytics" subtitle="Projected score, confidence calibration, timing profile, and miss taxonomy.">
      <div className="grid gap-3 md:grid-cols-4">
        <Card><p className="text-sm text-slate-500">Projected score</p><p className="mt-2 text-2xl font-bold">{overview.data.prediction ? `${overview.data.prediction.low}-${overview.data.prediction.high}` : "N/A"}</p></Card>
        <Card><p className="text-sm text-slate-500">Avg confidence</p><p className="mt-2 text-2xl font-bold">{overview.data.confidenceVsAccuracy.avgConfidence}/5</p></Card>
        <Card><p className="text-sm text-slate-500">Avg time</p><p className="mt-2 text-2xl font-bold">{overview.data.timingVsAccuracy.avgTimeSeconds}s</p></Card>
        <Card><p className="text-sm text-slate-500">Accuracy</p><p className="mt-2 text-2xl font-bold">{overview.data.confidenceVsAccuracy.accuracy}%</p></Card>
      </div>

      <Card>
        <h2 className="font-semibold">Content vs reasoning vs execution</h2>
        <p className="mt-2 text-sm text-slate-700">Content gaps: {overview.data.classifications.contentGaps} • Reasoning gaps: {overview.data.classifications.reasoningGaps} • Execution gaps: {overview.data.classifications.executionGaps}</p>
      </Card>

      <div className="grid gap-3 md:grid-cols-2">
        {trends.data.slice(-10).map((point) => <Card key={point.date}><p className="font-semibold">{point.date}</p><p className="text-sm text-slate-600">Accuracy {point.accuracy}% • Avg time {point.avgTime}s</p></Card>)}
      </div>
    </SectionPage>
  );
}
