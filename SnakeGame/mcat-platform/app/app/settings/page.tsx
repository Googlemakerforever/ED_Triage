"use client";

import { useState } from "react";
import { apiPatch } from "@/lib/api/client";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function SettingsPage() {
  const [targetScore, setTargetScore] = useState("520");
  const [testDate, setTestDate] = useState("");
  const [weeklyStudyHours, setWeeklyStudyHours] = useState("20");
  const [status, setStatus] = useState<string | null>(null);

  async function save() {
    await apiPatch("/api/profile", { targetScore: Number(targetScore), testDate, weeklyStudyHours: Number(weeklyStudyHours) });
    setStatus("Saved");
  }

  return (
    <SectionPage title="Profile & Settings" subtitle="Target score, test date, and study preferences.">
      <Card className="space-y-3 max-w-xl">
        <label className="text-sm">Target score<Input value={targetScore} onChange={(e) => setTargetScore(e.target.value)} /></label>
        <label className="text-sm">Test date<Input type="date" value={testDate} onChange={(e) => setTestDate(e.target.value)} /></label>
        <label className="text-sm">Weekly study hours<Input value={weeklyStudyHours} onChange={(e) => setWeeklyStudyHours(e.target.value)} /></label>
        <Button onClick={save}>Save settings</Button>
        {status && <p className="text-sm text-emerald-700">{status}</p>}
      </Card>
    </SectionPage>
  );
}
