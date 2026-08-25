import { SessionRunner } from "@/features/practice/session-runner";

export default function DiagnosticPage() {
  return (
    <SessionRunner
      title="Diagnostic Test"
      subtitle="Blueprint-mapped baseline assessment with timing, confidence, and miss classification (content/reasoning/execution)."
      startEndpoint="/api/diagnostic/start"
      submitEndpointTemplate="/api/diagnostic/:id/submit"
      defaultKind="diagnostic"
      lockKind
    />
  );
}
