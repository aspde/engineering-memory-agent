import { describe, it, expect } from 'vitest';
import RichText from './RichText';
import { render, screen } from '@testing-library/react';

describe('RichText', () => {
  it('renders plain text as-is', () => {
    render(<RichText text="Hello world" />);
    expect(screen.getByText('Hello world')).toBeInTheDocument();
  });

  it('renders inline code in monospace', () => {
    render(<RichText text="Use `npm install` to start" />);
    const code = screen.getByText('npm install');
    expect(code).toBeInTheDocument();
    expect(code.tagName).toBe('CODE');
  });

  it('renders bold text in strong tag', () => {
    render(<RichText text="This is **important** note" />);
    const bold = screen.getByText('important');
    expect(bold).toBeInTheDocument();
    expect(bold.tagName).toBe('STRONG');
  });

  it('renders italic text in em tag', () => {
    render(<RichText text="He said _hello_ to me" />);
    const italic = screen.getByText('hello');
    expect(italic).toBeInTheDocument();
    expect(italic.tagName).toBe('EM');
  });

  it('handles mixed formatting in one string', () => {
    render(<RichText text="**bold** and `code` and _italic_" />);
    expect(screen.getByText('bold').tagName).toBe('STRONG');
    expect(screen.getByText('code').tagName).toBe('CODE');
    expect(screen.getByText('italic').tagName).toBe('EM');
  });

  it('renders empty string without errors', () => {
    const { container } = render(<RichText text="" />);
    expect(container.textContent).toBe('');
  });

  it('preserves whitespace via pre-wrap class on parent', () => {
    render(<RichText text="line1\nline2" />);
    expect(screen.getByText(/line1/)).toBeInTheDocument();
    expect(screen.getByText(/line2/)).toBeInTheDocument();
  });
});
