"use client";

import { useState } from "react";
import { useApiGet } from "@/hooks/use-api-get";
import { apiPatch, apiPost } from "@/lib/api/client";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ErrorState, SkeletonCard, EmptyState } from "@/components/states/loaders";

type Task = { id: string; title: string; section: string; status: "todo" | "in_progress" | "done"; priority: number; dueDate: string };
type Plan = { id: string; tasks: Task[] };

export function StudyPlanView() {
  const [reloadKey, setReloadKey] = useState(0);
  const { data, loading, error } = useApiGet<Plan | null>(`/api/study-plan?reload=${reloadKey}`);

  async function generate() {
    await apiPost("/api/study-plan/generate", {});
    setReloadKey((x) => x + 1);
  }

  async function completeTask(id: string) {
    await apiPatch(`/api/study-tasks/${id}`, { status: "done" });
    setReloadKey((x) => x + 1);
  }

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;
  if (!data) return <EmptyState title="No study plan yet" body="Generate your first adaptive weekly plan." />;

  return (
    <SectionPage title="Study Plan" subtitle="Daily and weekly execution aligned to your target score." actions={<Button onClick={generate}>Regenerate week</Button>}>
      <div className="grid gap-3">
        {data.tasks.map((task) => (
          <Card key={task.id} className="flex items-center justify-between">
            <div>
              <p className="font-semibold">{task.title}</p>
              <p className="text-sm text-slate-600">{task.section} • Priority {task.priority}</p>
            </div>
            <Button variant={task.status === "done" ? "secondary" : "primary"} onClick={() => completeTask(task.id)}>
              {task.status === "done" ? "Completed" : "Mark done"}
            </Button>
          </Card>
        ))}
      </div>
    </SectionPage>
  );
}
