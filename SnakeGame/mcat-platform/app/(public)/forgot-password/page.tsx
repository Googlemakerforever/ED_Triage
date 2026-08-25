import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function ForgotPasswordPage() {
  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <Card>
        <h1 className="text-2xl font-bold">Forgot password</h1>
        <p className="mt-2 text-sm text-slate-600">Enter your email to receive a reset link.</p>
        <form className="mt-4 space-y-3">
          <label className="text-sm">Email<Input type="email" required /></label>
          <Button className="w-full">Send reset link</Button>
        </form>
      </Card>
    </div>
  );
}
