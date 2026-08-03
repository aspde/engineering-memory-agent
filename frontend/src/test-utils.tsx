import { type ReactNode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render, type RenderOptions } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppProvider } from './context/AppContext';

export function renderWithProviders(
  ui: ReactNode,
  options: { route?: string } & Omit<RenderOptions, 'wrapper'> = {},
) {
  const { route = '/', ...renderOptions } = options;

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter initialEntries={[route]}>
        <AppProvider>{children}</AppProvider>
      </MemoryRouter>
    );
  }

  return {
    user: userEvent.setup(),
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
  };
}

export { userEvent };
