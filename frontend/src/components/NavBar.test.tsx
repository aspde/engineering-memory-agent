import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import NavBar from './NavBar';

/**
 * Probe helper mock: breadth-layer endpoints (connectors, patrol) answer 404
 * by default (feature disabled), everything else 200.  Tests override this
 * per-case to assert the keep-item behaviour.
 */
let breadthStatus = 404;

beforeEach(() => {
  breadthStatus = 404;
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith('/api/connectors') || url.endsWith('/api/patrol/logs')) {
      return new Response('Not Found', { status: breadthStatus });
    }
    return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } });
  }));
});

function renderNav(initialEntries: string[] = ['/']) {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <NavBar />
    </MemoryRouter>,
  );
}

describe('NavBar', () => {
  it('renders the core nav items', () => {
    renderNav();
    expect(screen.getByLabelText('对话')).toBeInTheDocument();
    expect(screen.getByLabelText('记忆库')).toBeInTheDocument();
    expect(screen.getByLabelText('实体图谱')).toBeInTheDocument();
    expect(screen.getByLabelText('冲突')).toBeInTheDocument();
  });

  it('hides breadth-layer items when their probe returns 404', async () => {
    renderNav();
    await waitFor(() => {
      expect(screen.queryByLabelText('连接器')).not.toBeInTheDocument();
      expect(screen.queryByLabelText('巡检日志')).not.toBeInTheDocument();
    });
  });

  it('keeps breadth-layer items when the endpoint is available', async () => {
    breadthStatus = 200;
    renderNav();
    expect(await screen.findByLabelText('连接器')).toBeInTheDocument();
    expect(screen.getByLabelText('巡检日志')).toBeInTheDocument();
  });

  it('keeps breadth-layer items when the probe fails with a network error', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new Error('Network error');
    }));
    renderNav();
    expect(await screen.findByLabelText('连接器')).toBeInTheDocument();
    expect(screen.getByLabelText('巡检日志')).toBeInTheDocument();
  });

  it('marks the active route with aria-current="page"', () => {
    renderNav(['/memories']);
    const memBtn = screen.getByLabelText('记忆库');
    expect(memBtn).toHaveAttribute('aria-current', 'page');

    // Other items should not be active
    expect(screen.getByLabelText('对话')).not.toHaveAttribute('aria-current');
  });

  it('treats root path as active only for the 对话 button', () => {
    renderNav(['/']);
    expect(screen.getByLabelText('对话')).toHaveAttribute('aria-current', 'page');
    expect(screen.getByLabelText('记忆库')).not.toHaveAttribute('aria-current');
  });

  it('navigates on click', async () => {
    const user = userEvent.setup();
    renderNav(['/']);
    await user.click(screen.getByLabelText('记忆库'));
    // After click, 记忆库 should be the active one
    // (MemoryRouter updates location; the component re-renders)
    expect(screen.getByLabelText('记忆库')).toHaveAttribute('aria-current', 'page');
  });
});
