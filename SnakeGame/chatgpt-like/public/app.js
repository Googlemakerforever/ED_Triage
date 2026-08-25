const form = document.getElementById('chatForm');
const promptInput = document.getElementById('prompt');
const chat = document.getElementById('chat');

const MODEL = 'gpt-4.1-mini';

const messages = [
  {
    role: 'system',
    content:
      'You are a helpful, concise assistant. Ask clarifying questions only when needed. Be accurate and explicit about uncertainty.'
  }
];

function addMessage(role, text, extraClass = '') {
  const div = document.createElement('div');
  div.className = `msg ${role} ${extraClass}`.trim();
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  addMessage('user', prompt);
  messages.push({ role: 'user', content: prompt });
  promptInput.value = '';

  const waiting = document.createElement('div');
  waiting.className = 'msg assistant';
  waiting.textContent = 'Thinking...';
  chat.appendChild(waiting);
  chat.scrollTop = chat.scrollHeight;

  const button = form.querySelector('button');
  button.disabled = true;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, messages, temperature: 0.7 })
    });

    const data = await res.json();
    waiting.remove();

    if (!res.ok) {
      addMessage('assistant', data.error?.message || data.error || 'Request failed', 'error');
      return;
    }

    const reply = data.choices?.[0]?.message?.content || 'No response.';
    messages.push({ role: 'assistant', content: reply });
    addMessage('assistant', reply);
  } catch (err) {
    waiting.remove();
    addMessage('assistant', err.message, 'error');
  } finally {
    button.disabled = false;
    promptInput.focus();
  }
});
