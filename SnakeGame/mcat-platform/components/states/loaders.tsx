export function SkeletonCard() {
  return <div className="h-28 animate-pulse rounded-2xl border border-slate-200 bg-slate-100" />;
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="card-surface p-8 text-center">
      <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 text-sm text-slate-600">{body}</p>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
      {message}
    </div>
  );
}
