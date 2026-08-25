import { SessionRunner } from "@/features/practice/session-runner";

export default function PracticePage() {
  return (
    <SessionRunner
      title="Practice Hub"
      subtitle="Launch question or passage sessions with timing, confidence capture, and persisted review outputs."
      startEndpoint="/api/practice/sessions"
      submitEndpointTemplate="/api/practice/sessions/:id/submit"
      defaultKind="question"
    />
  );
}
