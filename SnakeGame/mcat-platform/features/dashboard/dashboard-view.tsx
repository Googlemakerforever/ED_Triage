"use client";

import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ErrorState, SkeletonCard, EmptyState } from "@/components/states/loaders";
import { SectionPage } from "@/components/layout/section-page";
import { useApiGet } from "@/hooks/use-api-get";

type DashboardData = {
  predictedScoreRange: string;
  sectionSnapshot: { accuracy: number; masteryAvg: number; avgConfidence: number };
  todaysPlan: { id: string; title: string; dueDate: string }[];
  weakestTopics: { topic: string; mastery: number }[];
  dueReviewItems: number;
  recommendedAction: string;
  diagnostic: { scorePercent: number; totalItems: number; completedAt: string } | null;
  classifications: { contentGaps: number; reasoningGaps: number; executionGaps: number };
};

export function DashboardView() {
  const { data, loading, error } = useApiGet<DashboardData>("/api/dashboard");

  if (loading) {
    return <div className="grid gap-4 md:grid-cols-3"><SkeletonCard /><SkeletonCard /><SkeletonCard /></div>;
  }
  if (error) return <ErrorState message={error} />;
  if (!data) return <EmptyState title="No dashboard data" body="Complete onboarding and start practice to generate analytics." />;

  return (
    <SectionPage
      title="Dashboard"
      subtitle="Your score trajectory and what to execute next."
      actions={!data.diagnostic ? <Link href="/app/diagnostic"><Button>Run Diagnostic</Button></Link> : undefined}
    >
      <div className="grid gap-4 md:grid-cols-4">
        <Card><p className="text-sm text-slate-500">Predicted score</p><p className="mt-2 text-3xl font-bold">{data.predictedScoreRange}</p></Card>
        <Card><p className="text-sm text-slate-500">Accuracy</p><p className="mt-2 text-3xl font-bold">{data.sectionSnapshot.accuracy}%</p></Card>
        <Card><p className="text-sm text-slate-500">Avg confidence</p><p className="mt-2 text-3xl font-bold">{data.sectionSnapshot.avgConfidence}/5</p></Card>
        <Card><p className="text-sm text-slate-500">Due reviews</p><p className="mt-2 text-3xl font-bold">{data.dueReviewItems}</p></Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <h2 className="text-lg font-semibold">Latest diagnostic</h2>
          {data.diagnostic ? (
            <p className="mt-2 text-sm text-slate-700">
              {data.diagnostic.scorePercent}% on {data.diagnostic.totalItems} items
            </p>
          ) : (
            <p className="mt-2 text-sm text-slate-700">No diagnostic completed yet.</p>
          )}
        </Card>
        <Card>
          <h2 className="text-lg font-semibold">Miss classification</h2>
          <p className="mt-2 text-sm text-slate-700">Content: {data.classifications.contentGaps} • Reasoning: {data.classifications.reasoningGaps} • Execution: {data.classifications.executionGaps}</p>
        </Card>
      </div>

      <Card>
        <h2 className="text-lg font-semibold">Recommended next action</h2>
        <p className="mt-2 text-sm text-slate-700">{data.recommendedAction}</p>
      </Card>
      <Card>
        <h2 className="text-lg font-semibold">Today&apos;s study plan</h2>
        <ul className="mt-3 space-y-2 text-sm text-slate-700">
          {data.todaysPlan.map((task) => <li key={task.id}>{task.title}</li>)}
        </ul>
      </Card>
    </SectionPage>
  );
}
