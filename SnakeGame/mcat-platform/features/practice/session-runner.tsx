"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { apiPost } from "@/lib/api/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { ErrorState } from "@/components/states/loaders";

type SessionItem = {
  sessionItemId: string;
  orderIndex: number;
  section: "CP" | "CARS" | "BB" | "PS";
  blueprintCategory: string | null;
  reasoningSkill: string | null;
  question: {
    id: string;
    stem: string;
    topic: string;
    difficulty: number;
    answerChoices: { id: string; label: string; text: string }[];
  } | null;
};

type StartedSession = {
  id: string;
  kind: string;
  timed: boolean;
  totalItems: number;
  items: SessionItem[];
};

type SubmitResult = {
  sessionId: string;
  submitted: boolean;
  scorePercent: number;
  totalCorrect: number;
  totalAnswered: number;
  totalElapsedSeconds: number;
  classifications: {
    contentGaps: number;
    reasoningGaps: number;
    executionGaps: number;
  };
};

type Props = {
  title: string;
  subtitle: string;
  startEndpoint: string;
  submitEndpointTemplate: string;
  defaultKind: "diagnostic" | "question" | "passage";
  lockKind?: boolean;
};

export function SessionRunner({
  title,
  subtitle,
  startEndpoint,
  submitEndpointTemplate,
  defaultKind,
  lockKind = false
}: Props) {
  const [kind, setKind] = useState<Props["defaultKind"]>(defaultKind);
  const [section, setSection] = useState<"CP" | "CARS" | "BB" | "PS">("CP");
  const [itemCount, setItemCount] = useState("16");
  const [session, setSession] = useState<StartedSession | null>(null);
  const [index, setIndex] = useState(0);
  const [selectedChoiceId, setSelectedChoiceId] = useState<string | null>(null);
  const [confidence, setConfidence] = useState("3");
  const [startedAt, setStartedAt] = useState<number>(Date.now());
  const [answers, setAnswers] = useState<Record<string, { selectedChoiceId: string; confidence: number; elapsedSeconds: number }>>({});
  const [result, setResult] = useState<SubmitResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const current = session?.items[index] ?? null;

  const progress = useMemo(() => {
    if (!session?.totalItems) return 0;
    return Math.round(((index + 1) / session.totalItems) * 100);
  }, [index, session?.totalItems]);

  async function start() {
    try {
      setError(null);
      setResult(null);
      const payload = {
        kind,
        section: kind === "diagnostic" ? undefined : section,
        timed: true,
        itemCount: Number(itemCount)
      };
      const started = await apiPost<typeof payload, StartedSession>(startEndpoint, payload);
      setSession(started);
      setIndex(0);
      setSelectedChoiceId(null);
      setConfidence("3");
      setAnswers({});
      setStartedAt(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to start session");
    }
  }

  function saveCurrentAnswer() {
    if (!current || !selectedChoiceId) return false;
    const elapsedSeconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    setAnswers((prev) => ({
      ...prev,
      [current.sessionItemId]: {
        selectedChoiceId,
        confidence: Number(confidence),
        elapsedSeconds
      }
    }));
    return true;
  }

  function next() {
    const ok = saveCurrentAnswer();
    if (!ok) {
      setError("Choose an answer before continuing.");
      return;
    }
    setError(null);
    setIndex((v) => Math.min((session?.totalItems ?? 1) - 1, v + 1));
    setSelectedChoiceId(null);
    setConfidence("3");
    setStartedAt(Date.now());
  }

  async function submit() {
    if (!session) return;
    const ok = saveCurrentAnswer();
    if (!ok) {
      setError("Choose an answer before submitting.");
      return;
    }

    try {
      const payload = {
        answers: Object.entries({ ...answers, [current!.sessionItemId]: {
          selectedChoiceId: selectedChoiceId!,
          confidence: Number(confidence),
          elapsedSeconds: Math.max(1, Math.round((Date.now() - startedAt) / 1000))
        } }).map(([sessionItemId, value]) => ({
          sessionItemId,
          selectedChoiceId: value.selectedChoiceId,
          confidence: value.confidence,
          elapsedSeconds: value.elapsedSeconds
        }))
      };

      const submitEndpoint = submitEndpointTemplate.replace(":id", session.id);
      const submitted = await apiPost<typeof payload, SubmitResult>(submitEndpoint, payload);
      setResult(submitted);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to submit session");
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <h1 className="text-2xl font-bold">{title}</h1>
        <p className="text-sm text-slate-600">{subtitle}</p>
        {!session && (
          <div className="mt-4 grid gap-3 md:grid-cols-4">
            <label className="text-sm">Mode
              <Select value={kind} onChange={(e) => setKind(e.target.value as Props["defaultKind"])} disabled={lockKind}>
                <option value="diagnostic">Diagnostic</option>
                <option value="question">Question Practice</option>
                <option value="passage">Passage Practice</option>
              </Select>
            </label>
            <label className="text-sm">Section
              <Select value={section} onChange={(e) => setSection(e.target.value as "CP" | "CARS" | "BB" | "PS")} disabled={kind === "diagnostic"}>
                <option value="CP">CP</option>
                <option value="CARS">CARS</option>
                <option value="BB">BB</option>
                <option value="PS">PS</option>
              </Select>
            </label>
            <label className="text-sm">Items
              <Input type="number" min={5} max={59} value={itemCount} onChange={(e) => setItemCount(e.target.value)} />
            </label>
            <div className="self-end"><Button onClick={start}>Start session</Button></div>
          </div>
        )}
      </Card>

      {error && <ErrorState message={error} />}

      {session && !result && current?.question && (
        <Card>
          <div className="mb-3 h-2 rounded bg-slate-100"><div className="h-2 rounded bg-brand-500" style={{ width: `${progress}%` }} /></div>
          <p className="text-xs uppercase tracking-wide text-slate-500">{current.section} • {current.blueprintCategory ?? "Blueprint"} • {current.reasoningSkill ?? "Reasoning"}</p>
          <p className="mt-2 text-lg font-semibold">{current.question.stem}</p>
          <p className="mt-1 text-sm text-slate-600">{current.question.topic} • Difficulty {current.question.difficulty}</p>

          <div className="mt-4 space-y-2">
            {current.question.answerChoices.map((choice) => (
              <button
                key={choice.id}
                className={`focus-ring w-full rounded-lg border px-3 py-2 text-left text-sm ${selectedChoiceId === choice.id ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-white"}`}
                onClick={() => setSelectedChoiceId(choice.id)}
              >
                <span className="font-semibold">{choice.label}.</span> {choice.text}
              </button>
            ))}
          </div>

          <div className="mt-4 flex items-end gap-3">
            <label className="text-sm">Confidence (1-5)
              <Select value={confidence} onChange={(e) => setConfidence(e.target.value)}>
                <option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option>
              </Select>
            </label>
            {index < session.totalItems - 1 ? (
              <Button onClick={next}>Save & Next</Button>
            ) : (
              <Button onClick={submit}>Submit Session</Button>
            )}
          </div>
        </Card>
      )}

      {result && (
        <Card>
          <h2 className="text-xl font-bold">Session Complete</h2>
          <p className="mt-2 text-sm text-slate-700">Score: <span className="font-semibold">{result.scorePercent}%</span> ({result.totalCorrect}/{result.totalAnswered})</p>
          <p className="text-sm text-slate-700">Total time: {Math.round(result.totalElapsedSeconds / 60)} min</p>
          <div className="mt-3 grid gap-2 md:grid-cols-3 text-sm">
            <div className="rounded border border-slate-200 p-3">Content gaps: <span className="font-semibold">{result.classifications.contentGaps}</span></div>
            <div className="rounded border border-slate-200 p-3">Reasoning gaps: <span className="font-semibold">{result.classifications.reasoningGaps}</span></div>
            <div className="rounded border border-slate-200 p-3">Execution gaps: <span className="font-semibold">{result.classifications.executionGaps}</span></div>
          </div>
          <div className="mt-4 flex gap-2">
            <Link href={`/app/review?sessionId=${result.sessionId}`}><Button>Open Review</Button></Link>
            <Link href="/app/dashboard"><Button variant="secondary">Back to Dashboard</Button></Link>
          </div>
        </Card>
      )}
    </div>
  );
}
