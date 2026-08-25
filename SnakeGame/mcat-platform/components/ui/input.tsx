import { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn("focus-ring w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm")} {...props} />;
}
