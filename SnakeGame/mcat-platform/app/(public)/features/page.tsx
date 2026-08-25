import { Card } from "@/components/ui/card";

const features = [
  ["Diagnostic Intelligence", "Blueprint-level baseline with content/reasoning/execution classification and confidence calibration."],
  ["Adaptive Planner", "Weekly/daily task generation that reprioritizes by weak skills, timing burden, and test date."],
  ["Practice + Review Engine", "Question and passage sessions with confidence, timing, and deep post-question logic review."],
  ["Error Log + Spaced Repetition", "Automatic conversion of misses into recurring-pattern logs and due-card queues."],
  ["Analytics", "Projected score bands, confidence vs accuracy, timing trends, and classification-level progress."],
  ["AI Tutor", "Mode-based tutoring for Teach, Hint, Socratic, Review, Drill, CARS Coach, and Planning Coach sessions."]
] as const;

export default function FeaturesPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16">
      <h1 className="text-3xl font-bold">Platform Features</h1>
      <p className="mt-3 text-slate-600">AegisMCAT is built for high-stakes execution, not passive content browsing.</p>
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {features.map(([title, body]) => (
          <Card key={title}>
            <h2 className="text-lg font-semibold">{title}</h2>
            <p className="mt-2 text-sm text-slate-700">{body}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}
