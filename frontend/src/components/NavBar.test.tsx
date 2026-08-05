import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import NavBar from './NavBar';

describe('NavBar', () => {
  it('renders all 5 nav items', () => {
    render(
      <MemoryRouter>
        <NavBar />
      </MemoryRouter>,
    );
    expect(screen.getByLabelText('对话')).toBeInTheDocument();
    expect(screen.getByLabelText('记忆库')).toBeInTheDocument();
    expect(screen.getByLabelText('实体图谱')).toBeInTheDocument();
    expect(screen.getByLabelText('连接器')).toBeInTheDocument();
    expect(screen.getByLabelText('巡检日志')).toBeInTheDocument();
  });

  it('marks the active route with aria-current="page"', () => {
    render(
      <MemoryRouter initialEntries={['/memories']}>
        <NavBar />
      </MemoryRouter>,
    );
    const memBtn = screen.getByLabelText('记忆库');
    expect(memBtn).toHaveAttribute('aria-current', 'page');

    // Other items should not be active
    expect(screen.getByLabelText('对话')).not.toHaveAttribute('aria-current');
  });

  it('treats root path as active only for the 对话 button', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <NavBar />
      </MemoryRouter>,
    );
    expect(screen.getByLabelText('对话')).toHaveAttribute('aria-current', 'page');
    expect(screen.getByLabelText('记忆库')).not.toHaveAttribute('aria-current');
  });

  it('navigates on click', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/']}>
        <NavBar />
      </MemoryRouter>,
    );
    await user.click(screen.getByLabelText('巡检日志'));
    // After click, 巡检日志 should be the active one
    // (MemoryRouter updates location; the component re-renders)
    expect(screen.getByLabelText('巡检日志')).toHaveAttribute('aria-current', 'page');
  });
});
