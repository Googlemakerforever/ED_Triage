import { SkeletonCard } from "@/components/states/loaders";

export default function Loading() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <SkeletonCard />
      <SkeletonCard />
      <SkeletonCard />
    </div>
  );
}
