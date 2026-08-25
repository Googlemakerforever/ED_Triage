import { PublicNav } from "@/components/layout/public-nav";

export default function PublicLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <PublicNav />
      <main>{children}</main>
    </div>
  );
}
