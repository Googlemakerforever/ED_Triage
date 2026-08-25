import { SessionRunner } from "@/features/practice/session-runner";

export default function PassagesPage() {
  return (
    <SessionRunner
      title="Passage Practice"
      subtitle="Run passage-style sets with timing and confidence capture, then review reasoning and execution misses."
      startEndpoint="/api/practice/sessions"
      submitEndpointTemplate="/api/practice/sessions/:id/submit"
      defaultKind="passage"
    />
  );
}
