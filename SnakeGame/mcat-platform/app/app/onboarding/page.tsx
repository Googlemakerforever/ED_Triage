"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiPost } from "@/lib/api/client";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

const sections = ["CP", "CARS", "BB", "PS"] as const;

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [targetScore, setTargetScore] = useState("520");
  const [testDate, setTestDate] = useState("");
  const [diagnosticScore, setDiagnosticScore] = useState("505");
  const [strongestSection, setStrongestSection] = useState("CP");
  const [weakestSection, setWeakestSection] = useState("CARS");
  const [weeklyHours, setWeeklyHours] = useState("20");
  const [studyPhase, setStudyPhase] = useState("mixed");

  async function submit() {
    await apiPost("/api/onboarding", {
      targetScore: Number(targetScore),
      testDate,
      diagnosticScore: Number(diagnosticScore),
      strongestSection,
      weakestSection,
      weeklyHours: Number(weeklyHours),
      studyPhase
    });
    router.push("/app/dashboard");
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <Card>
        <h1 className="text-2xl font-bold">Onboarding</h1>
        <p className="mt-1 text-sm text-slate-600">Step {step} of 3</p>
        {step === 1 && (
          <div className="mt-4 space-y-3">
            <label className="text-sm">Target score<Input value={targetScore} onChange={(e) => setTargetScore(e.target.value)} /></label>
            <label className="text-sm">Test date<Input type="date" value={testDate} onChange={(e) => setTestDate(e.target.value)} /></label>
            <label className="text-sm">Diagnostic score<Input value={diagnosticScore} onChange={(e) => setDiagnosticScore(e.target.value)} /></label>
            <Button onClick={() => setStep(2)}>Next</Button>
          </div>
        )}
        {step === 2 && (
          <div className="mt-4 space-y-3">
            <label className="text-sm">Strongest section<Select value={strongestSection} onChange={(e) => setStrongestSection(e.target.value)}>{sections.map((s) => <option key={s}>{s}</option>)}</Select></label>
            <label className="text-sm">Weakest section<Select value={weakestSection} onChange={(e) => setWeakestSection(e.target.value)}>{sections.map((s) => <option key={s}>{s}</option>)}</Select></label>
            <div className="flex gap-2"><Button variant="secondary" onClick={() => setStep(1)}>Back</Button><Button onClick={() => setStep(3)}>Next</Button></div>
          </div>
        )}
        {step === 3 && (
          <div className="mt-4 space-y-3">
            <label className="text-sm">Weekly study hours<Input value={weeklyHours} onChange={(e) => setWeeklyHours(e.target.value)} /></label>
            <label className="text-sm">Study phase<Select value={studyPhase} onChange={(e) => setStudyPhase(e.target.value)}><option value="foundation">Foundation</option><option value="mixed">Mixed</option><option value="intensive">Intensive</option><option value="final_review">Final review</option></Select></label>
            <div className="flex gap-2"><Button variant="secondary" onClick={() => setStep(2)}>Back</Button><Button onClick={submit}>Finish onboarding</Button></div>
          </div>
        )}
      </Card>
    </div>
  );
}
