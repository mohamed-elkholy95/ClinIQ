/**
 * Tests for the Sidebar navigation component.
 *
 * Verifies all 23 navigation links are rendered, the ClinIQ logo and
 * subtitle are present, and the close button triggers the callback.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Sidebar } from '../../components/Sidebar';

function renderSidebar(props: { isOpen?: boolean; onClose?: () => void } = {}) {
  const { isOpen = true, onClose = vi.fn() } = props;
  return {
    onClose,
    ...render(
      <BrowserRouter>
        <Sidebar isOpen={isOpen} onClose={onClose} />
      </BrowserRouter>
    ),
  };
}

describe('Sidebar — branding', () => {
  it('renders the ClinIQ logo text', () => {
    renderSidebar();
    expect(screen.getByText('ClinIQ')).toBeInTheDocument();
  });

  it('renders the subtitle', () => {
    renderSidebar();
    expect(screen.getByText('Clinical NLP')).toBeInTheDocument();
  });
});

describe('Sidebar — navigation items', () => {
  const expectedLinks = [
    { label: 'Dashboard', path: '/' },
    { label: 'Upload', path: '/upload' },
    { label: 'Entities', path: '/entities' },
    { label: 'ICD Codes', path: '/icd-codes' },
    { label: 'Summary', path: '/summary' },
    { label: 'Risk', path: '/risk' },
    { label: 'Medications', path: '/medications' },
    { label: 'Allergies', path: '/allergies' },
    { label: 'Vital Signs', path: '/vitals' },
    { label: 'Note Quality', path: '/quality' },
    { label: 'SDoH', path: '/sdoh' },
    { label: 'Comorbidity', path: '/comorbidity' },
    { label: 'De-identify', path: '/deidentify' },
    { label: 'Temporal', path: '/temporal' },
    { label: 'Assertions', path: '/assertions' },
    { label: 'Relations', path: '/relations' },
    { label: 'Classify', path: '/classify' },
    { label: 'Pipeline Explorer', path: '/pipeline' },
    { label: 'Search', path: '/search' },
    { label: 'Drift Monitor', path: '/drift' },
    { label: 'Streaming', path: '/stream' },
    { label: 'Timeline', path: '/timeline' },
    { label: 'Models', path: '/models' },
  ];

  it(`renders all ${expectedLinks.length} navigation links`, () => {
    renderSidebar();
    for (const { label } of expectedLinks) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it.each(expectedLinks)('renders "$label" with correct href "$path"', ({ label, path }) => {
    renderSidebar();
    const link = screen.getByText(label).closest('a');
    expect(link).toHaveAttribute('href', path);
  });
});

describe('Sidebar — close button', () => {
  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    renderSidebar({ onClose });
    // The X button for mobile — find by its parent button role
    const buttons = screen.getAllByRole('button');
    // Close button should be the one in the sidebar header
    const closeBtn = buttons.find((btn) => btn.querySelector('svg.lucide-x'));
    expect(closeBtn).toBeTruthy();
    fireEvent.click(closeBtn!);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when overlay is clicked', () => {
    const onClose = vi.fn();
    const { container } = renderSidebar({ onClose });
    // The overlay div is the first child that covers the viewport
    const overlay = container.querySelector('.fixed.inset-0');
    if (overlay) {
      fireEvent.click(overlay);
      expect(onClose).toHaveBeenCalledTimes(1);
    }
  });
});
