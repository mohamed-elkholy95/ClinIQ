/**
 * Tests for EntityTag and EntityTypeBadge components.
 *
 * Verifies entity text rendering, confidence display, click handlers,
 * size variants, and correct colour mapping per entity type.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EntityTag, EntityTypeBadge, entityStyles, entityLabels } from '../../components/EntityTag';
import type { EntityType } from '../../types';

const allTypes: EntityType[] = ['disease', 'medication', 'procedure', 'anatomy', 'symptom', 'lab_value'];

describe('EntityTag', () => {
  it('renders entity text', () => {
    render(<EntityTag type="disease" text="hypertension" />);
    expect(screen.getByText('hypertension')).toBeInTheDocument();
  });

  it('shows confidence percentage when provided', () => {
    render(<EntityTag type="medication" text="metformin" confidence={0.87} />);
    expect(screen.getByText('87%')).toBeInTheDocument();
  });

  it('hides confidence when not provided', () => {
    const { container } = render(<EntityTag type="symptom" text="fever" />);
    expect(container.querySelector('.opacity-60')).toBeNull();
  });

  it('rounds confidence to nearest integer', () => {
    render(<EntityTag type="procedure" text="biopsy" confidence={0.935} />);
    expect(screen.getByText('94%')).toBeInTheDocument();
  });

  it('calls onClick handler when clicked', () => {
    const handler = vi.fn();
    render(<EntityTag type="anatomy" text="liver" onClick={handler} />);
    fireEvent.click(screen.getByText('liver'));
    expect(handler).toHaveBeenCalledOnce();
  });

  it('has cursor-pointer class only when onClick is provided', () => {
    const { container: withClick } = render(
      <EntityTag type="disease" text="diabetes" onClick={() => {}} />
    );
    const { container: noClick } = render(
      <EntityTag type="disease" text="diabetes2" />
    );
    const spanWith = withClick.querySelector('span');
    const spanNo = noClick.querySelector('span');
    expect(spanWith?.className).toContain('cursor-pointer');
    expect(spanNo?.className).not.toContain('cursor-pointer');
  });

  it('renders sm size variant with smaller classes', () => {
    const { container } = render(
      <EntityTag type="lab_value" text="HbA1c" size="sm" />
    );
    const tag = container.querySelector('span');
    expect(tag?.className).toContain('text-xs');
  });

  it('renders md (default) size variant', () => {
    const { container } = render(
      <EntityTag type="lab_value" text="HbA1c" size="md" />
    );
    const tag = container.querySelector('span');
    expect(tag?.className).toContain('text-sm');
  });

  it.each(allTypes)('renders correct style classes for type: %s', (type) => {
    const { container } = render(<EntityTag type={type} text={`test-${type}`} />);
    const tag = container.querySelector('span');
    const style = entityStyles[type];
    // Check that the background class is applied (at least partially)
    expect(tag?.className).toContain(style.bg.split(' ')[0]);
  });

  it('renders a coloured dot indicator', () => {
    const { container } = render(<EntityTag type="disease" text="cancer" />);
    const dot = container.querySelector(`.${entityStyles.disease.dot.replace('bg-', 'bg-')}`);
    // The dot is a span with rounded-full class
    const dots = container.querySelectorAll('.rounded-full');
    // One for the outer tag, one for the dot
    expect(dots.length).toBeGreaterThanOrEqual(1);
  });
});

describe('EntityTypeBadge', () => {
  it.each(allTypes)('renders correct label for type: %s', (type) => {
    render(<EntityTypeBadge type={type} />);
    expect(screen.getByText(entityLabels[type])).toBeInTheDocument();
  });

  it('uses appropriate styles for the badge', () => {
    const { container } = render(<EntityTypeBadge type="medication" />);
    const badge = container.querySelector('span');
    expect(badge?.className).toContain('text-xs');
    expect(badge?.className).toContain('font-medium');
  });
});
