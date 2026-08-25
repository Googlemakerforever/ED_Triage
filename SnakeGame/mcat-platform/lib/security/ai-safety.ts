const INJECTION_PATTERNS = [
  /ignore\s+previous\s+instructions/i,
  /reveal\s+system\s+prompt/i,
  /print\s+hidden\s+instructions/i,
  /developer\s+message/i,
  /jailbreak/i
];

export function sanitizeAiText(input: string) {
  return input.replace(/[\u0000-\u001f\u007f]/g, " ").trim().slice(0, 1200);
}

export function looksLikePromptInjection(input: string) {
  return INJECTION_PATTERNS.some((pattern) => pattern.test(input));
}
