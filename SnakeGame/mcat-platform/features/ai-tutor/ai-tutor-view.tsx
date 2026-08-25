"use client";

import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useApiGet } from "@/hooks/use-api-get";
import { apiPost } from "@/lib/api/client";
import { SectionPage } from "@/components/layout/section-page";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { ErrorState, SkeletonCard } from "@/components/states/loaders";

type TutorMode = "teach" | "hint" | "socratic" | "review" | "drill" | "cars-coach" | "plan";
type Message = { id: string; role: "user" | "assistant"; content: string; createdAt?: string };
type Conversation = { id: string; title: string; mode: string; messages: Message[] };
type ConversationCreateResponse = Conversation;
type MessageCreateResponse = {
  userMessage?: Message;
  message: Message;
  conversationId: string;
  conversationTitle?: string;
};

export function AiTutorView() {
  const { data, loading, error } = useApiGet<Conversation[]>("/api/ai/conversations");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState<TutorMode>("teach");
  const [sendError, setSendError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isCreatingConversation, setIsCreatingConversation] = useState(false);
  const [hasInitializedSelection, setHasInitializedSelection] = useState(false);
  const creatingConversationPromiseRef = useRef<Promise<string> | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!data) return;
    setConversations(data);
  }, [data]);

  useEffect(() => {
    if (!data) return;
    if (!hasInitializedSelection && !conversationId && data.length > 0) {
      setConversationId(data[0].id);
      setHasInitializedSelection(true);
    }
  }, [data, hasInitializedSelection, conversationId]);

  const selectedConversation = useMemo(
    () => conversations.find((c) => c.id === conversationId),
    [conversationId, conversations]
  );
  const messageCount = selectedConversation?.messages.length ?? 0;

  useEffect(() => {
    const node = chatScrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [conversationId, messageCount, isSending]);

  async function createNewConversation() {
    if (creatingConversationPromiseRef.current) {
      return creatingConversationPromiseRef.current;
    }
    setIsCreatingConversation(true);
    setSendError(null);
    const promise = (async () => {
      const conv = await apiPost<{ mode: TutorMode }, ConversationCreateResponse>("/api/ai/conversations", {
        mode
      });
      setConversations((prev) => [conv, ...prev.filter((item) => item.id !== conv.id)]);
      setConversationId(conv.id);
      setHasInitializedSelection(true);
      setMessage("");
      return conv.id;
    })();
    creatingConversationPromiseRef.current = promise;
    try {
      return await promise;
    } finally {
      creatingConversationPromiseRef.current = null;
      setIsCreatingConversation(false);
    }
  }

  async function ensureConversation() {
    if (conversationId) return conversationId;
    const id = await createNewConversation();
    if (!id) {
      throw new Error("Unable to create a new conversation.");
    }
    return id;
  }

  function pushOptimisticMessages(id: string, userText: string, tempUserId: string, tempAssistantId: string) {
    setConversations((prev) =>
      prev.map((conversation) =>
        conversation.id === id
          ? {
              ...conversation,
              messages: conversation.messages.concat([
                { id: tempUserId, role: "user", content: userText },
                { id: tempAssistantId, role: "assistant", content: "Thinking..." }
              ])
            }
          : conversation
      )
    );
  }

  function rollbackOptimisticMessages(id: string, tempUserId: string, tempAssistantId: string) {
    setConversations((prev) =>
      prev.map((conversation) =>
        conversation.id === id
          ? {
              ...conversation,
              messages: conversation.messages.filter((item) => item.id !== tempUserId && item.id !== tempAssistantId)
            }
          : conversation
      )
    );
  }

  function commitServerMessages(
    id: string,
    tempUserId: string,
    tempAssistantId: string,
    response: MessageCreateResponse,
    fallbackUserText: string
  ) {
    setConversations((prev) =>
      prev.map((conversation) =>
        conversation.id === id
          ? {
              ...conversation,
              title: response.conversationTitle ?? conversation.title,
              messages: conversation.messages
                .filter((item) => item.id !== tempUserId && item.id !== tempAssistantId)
                .concat([
                  response.userMessage ?? { id: `${tempUserId}-final`, role: "user", content: fallbackUserText },
                  response.message
                ])
            }
          : conversation
      )
    );
  }

  async function send() {
    const trimmed = message.trim();
    if (!trimmed || isSending) return;
    setIsSending(true);
    setSendError(null);
    const tempUserId = `tmp-user-${Date.now()}`;
    const tempAssistantId = `tmp-assistant-${Date.now()}`;
    let id: string | null = null;
    try {
      setMessage("");
      id = await ensureConversation();
      pushOptimisticMessages(id, trimmed, tempUserId, tempAssistantId);

      const response = await apiPost<{ message: string }, MessageCreateResponse>(`/api/ai/conversations/${id}/messages`, { message: trimmed });
      commitServerMessages(id, tempUserId, tempAssistantId, response, trimmed);
    } catch (err) {
      if (id) {
        rollbackOptimisticMessages(id, tempUserId, tempAssistantId);
      }
      setMessage(trimmed);
      setSendError(err instanceof Error ? err.message : "Unable to send message.");
    } finally {
      setIsSending(false);
    }
  }

  async function onCreateConversation() {
    try {
      await createNewConversation();
    } catch (err) {
      setSendError(err instanceof Error ? err.message : "Unable to create conversation.");
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void send();
  }

  function onInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "Enter") {
      event.preventDefault();
      void send();
    }
  }

  if (loading) return <SkeletonCard />;
  if (error) return <ErrorState message={error} />;

  return (
    <SectionPage title="AI Tutor" subtitle="Chat-based tutoring with saved history, adaptive guidance, and auto-generated conversation titles.">
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="md:col-span-1">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Conversations</h2>
            <Button
              variant="secondary"
              onClick={() => {
                void onCreateConversation();
              }}
              disabled={isSending || isCreatingConversation}
            >
              {isCreatingConversation ? "Creating..." : "New conversation"}
            </Button>
          </div>
          <label className="mt-3 block text-sm">Mode
            <Select value={mode} onChange={(e) => setMode(e.target.value as TutorMode)}>
              <option value="teach">Teach</option>
              <option value="hint">Hint</option>
              <option value="socratic">Socratic</option>
              <option value="review">Review</option>
              <option value="drill">Drill</option>
              <option value="cars-coach">CARS Coach</option>
              <option value="plan">Planning Coach</option>
            </Select>
          </label>
          <h3 className="mt-4 text-sm font-semibold text-slate-700">Recent sessions</h3>
          <ul className="mt-2 space-y-2 text-sm text-slate-700">
            {conversations.map((c) => (
              <li key={c.id}>
                <button
                  className={`focus-ring w-full rounded border p-2 text-left hover:bg-slate-50 ${
                    c.id === conversationId ? "border-brand-300 bg-brand-50" : "border-slate-200"
                  }`}
                  onClick={() => setConversationId(c.id)}
                >
                  {c.title}
                </button>
              </li>
            ))}
          </ul>
        </Card>
        <Card className="md:col-span-2">
          <h2 className="font-semibold">Tutor chat</h2>
          <p className="text-xs text-slate-500">Conversation: {selectedConversation?.title ?? "New conversation"}</p>
          <div ref={chatScrollRef} className="mt-3 min-h-28 max-h-[420px] space-y-2 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
            {(selectedConversation?.messages.length ?? 0) > 0 ? (
              selectedConversation?.messages.map((item) => (
                <div
                  key={item.id}
                  className={`rounded-lg p-2 whitespace-pre-wrap ${item.role === "assistant" ? "bg-white border border-slate-200" : "bg-brand-50 border border-brand-200"}`}
                >
                  <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                    {item.role === "assistant" ? "Tutor" : "You"}
                  </p>
                  <p>{item.content}</p>
                </div>
              ))
            ) : (
              <p>Ask a question to begin.</p>
            )}
          </div>
          {sendError && <p className="mt-2 text-sm text-red-700">{sendError}</p>}
          <form className="mt-3 flex gap-2" onSubmit={onSubmit}>
            <Input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={onInputKeyDown}
              placeholder="Explain elimination logic for CARS inference traps"
            />
            <Button type="submit" disabled={!message.trim() || isSending}>{isSending ? "Sending..." : "Send"}</Button>
          </form>
        </Card>
      </div>
    </SectionPage>
  );
}
