import { useRef, useState } from 'react';
import { useMemories } from '../hooks/useMemories';

interface IngestSectionProps {
  /** Called after a successful ingest so the caller can refresh stats. */
  onIngest: () => void;
}

interface ToastState {
  type: 'success' | 'error';
  message: string;
}

type Tab = 'text' | 'file';

const ACCEPTED_EXTENSIONS =
  '.txt,.md,.py,.js,.ts,.json,.yaml,.yml,.toml,.cfg,.ini,.sql,.html,.css,.sh,.java,.go,.rs,.c,.cpp,.h,.rb,.php';

const TOAST_DURATION_MS = 4000;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function IngestSection({ onIngest }: IngestSectionProps) {
  const { ingestText, ingestFile } = useMemories();

  const [tab, setTab] = useState<Tab>('text');
  const [text, setText] = useState('');
  const [docName, setDocName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [toast, setToast] = useState<ToastState | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  const showToast = (t: ToastState) => {
    setToast(t);
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToast(null);
      toastTimerRef.current = null;
    }, TOAST_DURATION_MS);
  };

  const handleIngestText = async () => {
    const content = text.trim();
    const documentId = docName.trim();
    if (!content || !documentId || isIngesting) return;
    setIsIngesting(true);
    try {
      const res = await ingestText(documentId, content);
      showToast({ type: 'success', message: `✅ 已摄入：${res.chunks_written} 个块` });
      setText('');
      setDocName('');
      onIngest();
    } catch (err) {
      showToast({
        type: 'error',
        message: `摄入失败: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setIsIngesting(false);
    }
  };

  const handleIngestFile = async () => {
    if (!file || isIngesting) return;
    setIsIngesting(true);
    try {
      const res = await ingestFile(file);
      showToast({
        type: 'success',
        message: `✅ 已摄入：${file.name} → ${res.chunks_written} 个块`,
      });
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      onIngest();
    } catch (err) {
      showToast({
        type: 'error',
        message: `摄入失败: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setIsIngesting(false);
    }
  };

  const canIngestText = text.trim().length > 0 && docName.trim().length > 0;

  return (
    <div>
      {/* Tabs */}
      <div className="mb-4 flex gap-1 border-b border-gray-200">
        {(['text', 'file'] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`rounded-t-lg border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              tab === t
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {t === 'text' ? '粘贴文本' : '上传文件'}
          </button>
        ))}
      </div>

      {tab === 'text' ? (
        <div className="space-y-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="粘贴要摄入的文档、代码或任意文本…"
            rows={8}
            className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none"
          />
          <input
            type="text"
            value={docName}
            onChange={(e) => setDocName(e.target.value)}
            placeholder="用于标识这段内容（如 README、app.py）"
            className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none"
          />
          <button
            type="button"
            onClick={handleIngestText}
            disabled={!canIngestText || isIngesting}
            className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isIngesting ? '正在分块、嵌入、入库…' : '摄入文本'}
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-gray-500 file:mr-3 file:rounded-lg file:border-0 file:bg-gray-100 file:px-3 file:py-2 file:text-sm file:font-medium file:text-gray-700 hover:file:bg-gray-200"
          />
          {file && (
            <p className="text-sm text-gray-500">
              已选择: <span className="font-mono">{file.name}</span> ({formatBytes(file.size)})
            </p>
          )}
          <button
            type="button"
            onClick={handleIngestFile}
            disabled={!file || isIngesting}
            className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isIngesting ? '正在分块、嵌入、入库…' : '摄入文件'}
          </button>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div
          className={`fixed bottom-4 right-4 z-50 rounded-lg px-4 py-3 text-sm font-medium text-white shadow-lg ${
            toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'
          }`}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}
