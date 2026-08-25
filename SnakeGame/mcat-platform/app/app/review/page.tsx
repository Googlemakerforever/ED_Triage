"use client";

import { useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { useApiGet } from "@/hooks/use-api-get";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { EmptyState, ErrorState, SkeletonCard } from "@/components/states/loaders";

type SessionReview = {
  sessionId: string;
  scorePercent: number;
  items: {
    sessionItemId: string;
    question: string;
    section: string;
    topic?: string;
    blueprintCategory?: string;
    reasoningSkill?: string;
    selectedAnswer: string;
    correctAnswer: string;
    explanation: string;
    whyWrong: string;
    mistakeType?: string;
    missClass?: string;
    confidence: number;
    elapsedSeconds: number;
    nextTimeRule: string;
  }[];
};

type ReviewPayload = {
  items: { id: string; summary: string; keyTakeaway: string; createdAt: string }[];
  recentSessions: { id: string; kind: string; scorePercent: number | null }[];
};

export default function ReviewPage() {
  const params = useSearchParams();
  const sessionId = params.get("sessionId");

  const reviewUrl = useMemo(() => {
    if (!sessionId) return null;
    return `/api/practice/sessions/${sessionId}/review`;
  }, [sessionId]);

  const sessionReview = useApiGet<SessionReview>(reviewUrl ?? "__disabled__");
  const fallbackReview = useApiGet<ReviewPayload>("/api/review");

  if (sessionId) {
    if (sessionReview.loading) return <SkeletonCard />;
    if (sessionReview.error) return <ErrorState message={sessionReview.error} />;
    if (!sessionReview.data?.items.length) return <EmptyState title="No review items" body="Complete a session to populate review analysis." />;

    return (
      <SectionPage title="Review Engine" subtitle={`Session ${sessionId} • score ${sessionReview.data.scorePercent}%`}>
        <div className="space-y-3">
          {sessionReview.data.items.map((item) => (
            <Card key={item.sessionItemId}>
              <p className="font-semibold">{item.question}</p>
              <p className="mt-1 text-xs uppercase text-slate-500">{item.section} • {item.blueprintCategory ?? "Blueprint"} • {item.reasoningSkill ?? "Reasoning"}</p>
              <p className="mt-2 text-sm"><span className="font-medium">Your answer:</span> {item.selectedAnswer}</p>
              <p className="mt-1 text-sm"><span className="font-medium">Correct answer:</span> {item.correctAnswer}</p>
              <p className="mt-1 text-sm"><span className="font-medium">Why it works:</span> {item.explanation}</p>
              <p className="mt-1 text-sm"><span className="font-medium">Why misses happen:</span> {item.whyWrong}</p>
              <p className="mt-1 text-sm"><span className="font-medium">Classification:</span> {item.missClass ?? "none"} {item.mistakeType ? `(${item.mistakeType})` : ""}</p>
              <p className="mt-1 text-sm"><span className="font-medium">Transfer rule:</span> {item.nextTimeRule}</p>
            </Card>
          ))}
        </div>
      </SectionPage>
    );
  }

  if (fallbackReview.loading) return <SkeletonCard />;
  if (fallbackReview.error) return <ErrorState message={fallbackReview.error} />;
  if (!fallbackReview.data?.items?.length) return <EmptyState title="No review history" body="Run practice or diagnostic sessions to populate the review engine." />;

  return (
    <SectionPage title="Review Engine" subtitle="Recent review entries from your missed questions.">
      {!!fallbackReview.data.recentSessions.length && (
        <Card>
          <h2 className="font-semibold">Recent submitted sessions</h2>
          <ul className="mt-2 space-y-1 text-sm text-slate-700">
            {fallbackReview.data.recentSessions.map((s) => (
              <li key={s.id}>Session {s.id.slice(-6)} • {s.kind} • {s.scorePercent ?? "N/A"}%</li>
            ))}
          </ul>
        </Card>
      )}
      <div className="space-y-3">
        {fallbackReview.data.items.map((item) => (
          <Card key={item.id}>
            <p className="font-semibold">{item.summary}</p>
            <p className="mt-1 text-sm text-slate-700"><span className="font-medium">Key takeaway:</span> {item.keyTakeaway}</p>
          </Card>
        ))}
      </div>
    </SectionPage>
  );
}
