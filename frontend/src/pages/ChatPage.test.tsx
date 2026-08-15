import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { getThreadMessages } from '../api/agent';
import { runScenario } from '../api/scenarios';
import { useChat } from '../hooks/useChat';
import ChatPage from './ChatPage';

// Mock the context so each test controls threadId / messages / interrupts.
vi.mock('../context/AppContext', () => ({
  useAppState: vi.fn(),
  useAppDispatch: vi.fn(),
}));

// Mock the API modules — the page wires them to the context, not to a server.
vi.mock('../api/agent', () => ({ getThreadMessages: vi.fn() }));
vi.mock('../api/scenarios', () => ({ runScenario: vi.fn() }));
vi.mock('../hooks/useChat', () => ({ useChat: vi.fn() }));

type Mock = ReturnType<typeof vi.fn>;

interface TestState {
  threadId: string;
  loadedThreadId: string | null;
  messages: unknown[];
  pendingInterrupt: unknown;
  waitingForApproval: boolean;
  activeScenario: string | null;
  isStreaming: boolean;
}

let state: TestState;
let dispatch: Mock;
let sendMessage: Mock;
let resume: Mock;

beforeEach(() => {
  vi.clearAllMocks();
  state = {
    threadId: 't-1',
    loadedThreadId: null,
    messages: [],
    pendingInterrupt: null,
    waitingForApproval: false,
    activeScenario: null,
    isStreaming: false,
  };
  (useAppState as unknown as Mock).mockReturnValue(state);
  dispatch = vi.fn();
  (useAppDispatch as unknown as Mock).mockReturnValue(dispatch);

  sendMessage = vi.fn().mockResolvedValue(null);
  resume = vi.fn();
  (useChat as unknown as Mock).mockReturnValue({ sendMessage, resume, isStreaming: false });

  // Default: the history fetch resolves with an empty conversation.
  (getThreadMessages as unknown as Mock).mockResolvedValue({
    thread_id: 't-1',
    messages: [],
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

function renderPage() {
  return render(
    <MemoryRouter>
      <ChatPage />
    </MemoryRouter>,
  );
}

describe('ChatPage', () => {
  it('shows the empty-state welcome and a functional input when no history loads', async () => {
    renderPage();
    // History loads async — the page briefly shows a loading state first.
    expect(await screen.findByText('EMA — Engineering Memory Agent')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('向 EMA 提问…')).toBeInTheDocument();
  });

  it('lazy-loads message history for the active thread and dispatches SET_MESSAGES', async () => {
    (getThreadMessages as unknown as Mock).mockResolvedValue({
      thread_id: 't-1',
      messages: [{ role: 'assistant', content: '历史回答内容' }],
    });

    renderPage();

    await waitFor(() => expect(getThreadMessages).toHaveBeenCalledWith('t-1'));
    await waitFor(() =>
      expect(dispatch).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'SET_MESSAGES' }),
      ),
    );
    expect(dispatch).toHaveBeenCalledWith({ type: 'SET_LOADED_THREAD', threadId: 't-1' });
  });

  it('falls back to an empty conversation when the history fetch 404s (new thread)', async () => {
    (getThreadMessages as unknown as Mock).mockRejectedValue({ status: 404 });

    renderPage();

    await waitFor(() => expect(dispatch).toHaveBeenCalledWith({ type: 'SET_MESSAGES', messages: [] }));
    // The page renders the empty-state welcome instead of crashing.
    expect(await screen.findByText('EMA — Engineering Memory Agent')).toBeInTheDocument();
  });

  it('renders loaded messages from state', async () => {
    state.messages = [
      { role: 'user', content: '你好' },
      { role: 'assistant', content: '有什么可以帮你？' },
    ];

    renderPage();

    expect(await screen.findByText('你好')).toBeInTheDocument();
    expect(await screen.findByText('有什么可以帮你？')).toBeInTheDocument();
  });

  it('sends a message through the chat hook and does not force-write by default', async () => {
    const user = userEvent.setup();
    renderPage();
    // Wait for the async history load to finish so the input is enabled.
    await screen.findByPlaceholderText('向 EMA 提问…');

    await user.type(screen.getByLabelText('聊天输入'), 'hello');
    await user.click(screen.getByRole('button', { name: '发送' }));

    expect(sendMessage).toHaveBeenCalledWith('hello', false);
  });

  it('force-writes the message and toasts the write outcome when "记住这条" is checked', async () => {
    sendMessage.mockResolvedValue({ action: 'inserted', summary: '端口改为 8080' });
    const user = userEvent.setup();
    renderPage();
    await screen.findByPlaceholderText('向 EMA 提问…');

    await user.click(screen.getByLabelText('记住这条'));
    await user.type(screen.getByLabelText('聊天输入'), '记住这个决定');
    await user.click(screen.getByRole('button', { name: '发送' }));

    // force-write is a single-action toggle: the flag travels with this send.
    expect(sendMessage).toHaveBeenCalledWith('记住这个决定', true);
    // The toast shows the distilled summary so the user can verify the write.
    expect(await screen.findByText('已写入：端口改为 8080')).toBeInTheDocument();
  });

  it('toasts the conflict outcome when the forced write hits a conflict', async () => {
    sendMessage.mockResolvedValue({ action: 'conflict', summary: '…' });
    const user = userEvent.setup();
    renderPage();
    await screen.findByPlaceholderText('向 EMA 提问…');

    await user.click(screen.getByLabelText('记住这条'));
    await user.type(screen.getByLabelText('聊天输入'), 'x');
    await user.click(screen.getByRole('button', { name: '发送' }));

    expect(await screen.findByText('检测到冲突，请在记忆库中处理')).toBeInTheDocument();
  });

  it('shows no write toast when the turn produced no memory write', async () => {
    // sendMessage resolves null (default): no write happened → no toast.
    const user = userEvent.setup();
    renderPage();
    await screen.findByPlaceholderText('向 EMA 提问…');

    await user.click(screen.getByLabelText('记住这条'));
    await user.type(screen.getByLabelText('聊天输入'), 'x');
    await user.click(screen.getByRole('button', { name: '发送' }));

    await waitFor(() => expect(sendMessage).toHaveBeenCalledWith('x', true));
    expect(screen.queryByText('已写入新记忆')).not.toBeInTheDocument();
    expect(screen.queryByText('记忆写入失败')).not.toBeInTheDocument();
  });

  it('disables the input and shows the streaming placeholder while isStreaming', async () => {
    state.messages = [{ role: 'assistant', content: '正在回答' }];
    (useChat as unknown as Mock).mockReturnValue({ sendMessage, resume, isStreaming: true });

    renderPage();

    // History loads async; once done the streaming placeholder takes over.
    expect(await screen.findByPlaceholderText('回复生成中…')).toBeInTheDocument();
    expect(screen.getByLabelText('聊天输入')).toBeDisabled();
  });

  it('auto-triggers the active scenario once for a fresh thread', async () => {
    (runScenario as unknown as Mock).mockResolvedValue({ result: '复盘结果内容' });
    state.activeScenario = 'postmortem';
    state.loadedThreadId = 't-1'; // history already loaded, so the trigger fires

    renderPage();

    await waitFor(() => expect(runScenario).toHaveBeenCalledWith('postmortem', {}, 't-1'));
    // Placeholder messages are added optimistically before the API returns.
    expect(dispatch).toHaveBeenCalledWith({ type: 'CLEAR_ACTIVE_SCENARIO' });
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'ADD_MESSAGE',
        message: expect.objectContaining({ content: '触发场景: postmortem' }),
      }),
    );
  });

  it('does not re-trigger a scenario when the thread already has messages', async () => {
    state.activeScenario = 'code_review';
    state.loadedThreadId = 't-1';
    state.messages = [{ role: 'user', content: '已有对话' }];

    renderPage();

    // Give the effect a tick to run; the trigger must stay off.
    await new Promise((r) => setTimeout(r, 0));
    expect(runScenario).not.toHaveBeenCalled();
  });
});
