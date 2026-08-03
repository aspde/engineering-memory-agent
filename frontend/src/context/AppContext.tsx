import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from 'react';
import type { AppAction, AppState, Message } from '../types';

const initialState: AppState = {
  threadId: crypto.randomUUID(),
  messages: [],
  pendingInterrupt: null,
  waitingForApproval: false,
  isStreaming: false,
  threads: [],
  threadsFetchedAt: 0,
  loadedThreadId: null,
  memFilterId: null,
};

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_THREAD_ID':
      return { ...state, threadId: action.threadId, messages: [], pendingInterrupt: null, waitingForApproval: false };
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.message] };
    case 'UPDATE_LAST_MESSAGE': {
      const { messages } = state;
      if (messages.length === 0) return state;
      const lastIndex = messages.length - 1;
      const last = messages[lastIndex];
      const updated: Message = {
        ...last,
        content: action.appendContent ? last.content + action.appendContent : last.content,
        _meta: action.meta ?? last._meta,
      };
      return { ...state, messages: [...messages.slice(0, lastIndex), updated] };
    }
    case 'SET_MESSAGES':
      return { ...state, messages: action.messages };
    case 'SET_INTERRUPT':
      return { ...state, pendingInterrupt: action.interrupt, waitingForApproval: true };
    case 'CLEAR_INTERRUPT':
      return { ...state, pendingInterrupt: null, waitingForApproval: false };
    case 'SET_STREAMING':
      return { ...state, isStreaming: action.isStreaming };
    case 'SET_THREADS':
      return { ...state, threads: action.threads, threadsFetchedAt: Date.now() };
    case 'SET_MEM_FILTER':
      return { ...state, memFilterId: action.memId };
    case 'CLEAR_MEM_FILTER':
      return { ...state, memFilterId: null };
    case 'NEW_CONVERSATION':
      return {
        ...state,
        threadId: action.threadId,
        messages: [],
        pendingInterrupt: null,
        waitingForApproval: false,
        isStreaming: false,
        loadedThreadId: null,
      };
    case 'SET_LOADED_THREAD':
      return { ...state, loadedThreadId: action.threadId };
    case 'INVALIDATE_THREADS':
      return { ...state, threadsFetchedAt: 0 };
    default:
      return state;
  }
}

interface AppContextValue {
  state: AppState;
  dispatch: Dispatch<AppAction>;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  return <AppContext.Provider value={{ state, dispatch }}>{children}</AppContext.Provider>;
}

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return ctx;
}

export function useAppState(): AppState {
  return useAppContext().state;
}

export function useAppDispatch(): Dispatch<AppAction> {
  return useAppContext().dispatch;
}
