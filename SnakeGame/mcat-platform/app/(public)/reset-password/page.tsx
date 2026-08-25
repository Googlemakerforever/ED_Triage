import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function ResetPasswordPage() {
  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <Card>
        <h1 className="text-2xl font-bold">Reset password</h1>
        <form className="mt-4 space-y-3">
          <label className="text-sm">New password<Input type="password" required /></label>
          <label className="text-sm">Confirm password<Input type="password" required /></label>
          <Button className="w-full">Update password</Button>
        </form>
      </Card>
    </div>
  );
}
